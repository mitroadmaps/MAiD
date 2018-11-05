import sys
sys.path.append('../lib')

from discoverlib import geom, graph
import maid_model as model
import model_utils
import tileloader as tileloader

import json
import numpy
import math
import os.path
from PIL import Image
import random
import scipy.ndimage
import sys
import tensorflow as tf
import time

MAX_PATH_LENGTH = 1000000
SEGMENT_LENGTH = 20
TILE_MODE = 'sat'
THRESHOLD_BRANCH = 0.2
THRESHOLD_FOLLOW = 0.2
WINDOW_SIZE = 256
SAVE_EXAMPLES = False
ANGLE_ONEHOT = 64
M6 = True
CACHE_M6 = True
FILTER_BY_TAG = None
EXISTING_GRAPH_FNAME = None

REGION = 'boston'
TILE_SIZE = 4096
TILE_START = geom.Point(1, -1).scale(TILE_SIZE)
TILE_END = geom.Point(3, 1).scale(TILE_SIZE)

USE_TL_LOCATIONS = True
MANUAL_RELATIVE = geom.Point(-1, -2).scale(TILE_SIZE)
MANUAL_POINT1 = geom.Point(2560, 522)
MANUAL_POINT2 = geom.Point(2592, 588)

def vector_to_action(extension_vertex, angle_outputs, threshold):
	# mask out buckets that are similar to existing edges
	blacklisted_buckets = set()
	for edge in extension_vertex.out_edges:
		angle = geom.Point(1, 0).signed_angle(edge.segment().vector())
		bucket = int((angle + math.pi) * 64.0 / math.pi / 2)
		for offset in xrange(6):
			clockwise_bucket = (bucket + offset) % 64
			counterclockwise_bucket = (bucket + 64 - offset) % 64
			blacklisted_buckets.add(clockwise_bucket)
			blacklisted_buckets.add(counterclockwise_bucket)

	seen_vertices = set()
	search_queue = []
	nearby_points = {}
	seen_vertices.add(extension_vertex)
	for edge in extension_vertex.out_edges:
		search_queue.append((graph.EdgePos(edge, 0), 0))
	while len(search_queue) > 0:
		edge_pos, distance = search_queue[0]
		search_queue = search_queue[1:]
		if distance > 0:
			nearby_points[edge_pos.point()] = distance
		if distance >= 4 * SEGMENT_LENGTH:
			continue

		edge = edge_pos.edge
		l = edge.segment().length()
		if edge_pos.distance + SEGMENT_LENGTH < l:
			search_queue.append((graph.EdgePos(edge, edge_pos.distance + SEGMENT_LENGTH), distance + SEGMENT_LENGTH))
		elif edge.dst not in seen_vertices:
			seen_vertices.add(edge.dst)
			for edge in edge.dst.out_edges:
				search_queue.append((graph.EdgePos(edge, 0), distance + l - edge_pos.distance))

	# any leftover targets above threshold?
	best_bucket = None
	best_value = None
	for bucket in xrange(64):
		if bucket in blacklisted_buckets:
			continue
		next_point = model_utils.get_next_point(extension_vertex.point, bucket, SEGMENT_LENGTH)
		bad = False
		for nearby_point, distance in nearby_points.items():
			if nearby_point.distance(next_point) < 0.5 * (SEGMENT_LENGTH + distance):
				bad = True
				break
		if bad:
			continue

		value = angle_outputs[bucket]
		if value > threshold and (best_bucket is None or value > best_value):
			best_bucket = bucket
			best_value = value

	x = numpy.zeros((64,), dtype='float32')
	if best_bucket is not None:
		x[best_bucket] = best_value
	return x

def eval(paths, m, session, max_path_length=MAX_PATH_LENGTH, segment_length=SEGMENT_LENGTH, save=False, compute_targets=True, max_batch_size=model.BATCH_SIZE, window_size=WINDOW_SIZE, verbose=True, threshold_override=None, cache_m6=None):
	angle_losses = []
	detect_losses = []
	losses = []
	path_lengths = {path_idx: 0 for path_idx in xrange(len(paths))}

	last_time = None
	big_time = None

	last_extended = False

	for len_it in xrange(99999999):
		if len_it % 500 == 0 and verbose:
			print 'it {}'.format(len_it)
			big_time = time.time()
		path_indices = []
		extension_vertices = []
		for path_idx in xrange(len(paths)):
			if path_lengths[path_idx] >= max_path_length:
				continue
			extension_vertex = paths[path_idx].pop()
			if extension_vertex is None:
				continue
			path_indices.append(path_idx)
			path_lengths[path_idx] += 1
			extension_vertices.append(extension_vertex)

			if len(path_indices) >= max_batch_size:
				break

		if len(path_indices) == 0:
			break

		batch_inputs = []
		batch_detect_targets = []
		batch_angle_targets = numpy.zeros((len(path_indices), 64), 'float32')
		inputs_per_path = 1

		for i in xrange(len(path_indices)):
			path_idx = path_indices[i]

			path_input, path_detect_target = model_utils.make_path_input(paths[path_idx], extension_vertices[i], segment_length, window_size=window_size)
			if type(path_input) == list:
				batch_inputs.extend([x[:, :, 0:3] for x in path_input])
				inputs_per_path = len(path_input)
			else:
				batch_inputs.append(path_input[:, :, 0:3])
			#batch_detect_targets.append(path_detect_target)
			batch_detect_targets.append(numpy.zeros((64, 64, 1), dtype='float32'))

			if compute_targets:
				angle_targets, _ = model_utils.compute_targets_by_best(paths[path_idx], extension_vertices[i], segment_length)
				batch_angle_targets[i, :] = angle_targets

		# run model
		if M6:
			angle_loss, detect_loss, loss = 0.0, 0.0, 0.0
			if cache_m6 is not None:
				p = extension_vertices[0].point.sub(paths[0].tile_data['rect'].start).scale(0.25)
				batch_angle_outputs = numpy.array([cache_m6[p.x, p.y, :]], dtype='float32')
			else:
				pre_outputs = session.run(m.outputs, feed_dict={
					m.is_training: False,
					m.inputs: batch_inputs,
				})
				batch_angle_outputs = pre_outputs[:, window_size/8, window_size/8, :]
			#batch_angle_outputs = numpy.zeros((len(path_indices), 64), dtype='float32')
			#for i in xrange(64):
			#	batch_angle_outputs[:, i] = pre_outputs[:, 32, 32, i/4]
		else:
			# DELETE ME INPUT RESIZE
			#batch_inputs = [scipy.ndimage.zoom(im, [0.5, 0.5, 1.0], order=0) for im in batch_inputs]
			#batch_inputs = [scipy.ndimage.zoom(im, [2.0, 2.0, 1.0], order=0) for im in batch_inputs]

			feed_dict = {
				m.is_training: False,
				m.inputs: batch_inputs,
				m.angle_targets: [x for x in batch_angle_targets for _ in xrange(inputs_per_path)],
				m.detect_targets: [x for x in batch_detect_targets for _ in xrange(inputs_per_path)],
			}
			if ANGLE_ONEHOT:
				feed_dict[m.angle_onehot] = model_utils.get_angle_onehot(ANGLE_ONEHOT)
			batch_angle_outputs, angle_loss, detect_loss, loss = session.run([m.angle_outputs, m.angle_loss, m.detect_loss, m.loss], feed_dict=feed_dict)

		if inputs_per_path > 1:
			actual_outputs = numpy.zeros((len(path_indices), 64), 'float32')
			for i in xrange(len(path_indices)):
				actual_outputs[i, :] = batch_angle_outputs[i*inputs_per_path:(i+1)*inputs_per_path, :].max(axis=0)
			batch_angle_outputs = actual_outputs

		angle_losses.append(angle_loss)
		losses.append(loss)

		if (save is True and len_it % 1 == 0) or (save == 'extends' and last_extended):
			fname = '/home/ubuntu/data/{}_'.format(len_it)
			save_angle_targets = batch_angle_targets[0, :]
			if not compute_targets:
				save_angle_targets = None
			model_utils.make_path_input(paths[path_indices[0]], extension_vertices[0], segment_length, fname=fname, angle_targets=save_angle_targets, angle_outputs=batch_angle_outputs[0, :], window_size=window_size)

			with open(fname + 'meta.txt', 'w') as f:
				f.write('max angle output: {}\n'.format(batch_angle_outputs[0, :].max()))

		for i in xrange(len(path_indices)):
			path_idx = path_indices[i]
			if len(extension_vertices[i].out_edges) >= 2:
				threshold = THRESHOLD_BRANCH
			else:
				threshold = THRESHOLD_FOLLOW
			if threshold_override is not None:
				threshold = threshold_override

			x = vector_to_action(extension_vertices[i], batch_angle_outputs[i, :], threshold)
			last_extended = x.max() > 0
			paths[path_idx].push(extension_vertices[i], x, segment_length, training=False, branch_threshold=0.01, follow_threshold=0.01)

	if save:
		paths[0].graph.save('out.graph')

	return numpy.mean(angle_losses), numpy.mean(detect_losses), numpy.mean(losses), len_it

def graph_filter(g, threshold=0.3, min_len=None):
	road_segments, _ = graph.get_graph_road_segments(g)
	bad_edges = set()
	for rs in road_segments:
		if min_len is not None and len(rs.edges) < min_len:
			bad_edges.update(rs.edges)
			continue
		probs = []
		if len(rs.edges) < 5 or True:
			for edge in rs.edges:
				if hasattr(edge, 'prob'):
					probs.append(edge.prob)
		else:
			for edge in rs.edges[2:-2]:
				if hasattr(edge, 'prob'):
					probs.append(edge.prob)
		if not probs:
			continue
		avg_prob = numpy.mean(probs)
		if avg_prob < threshold:
			bad_edges.update(rs.edges)
	print 'filtering {} edges'.format(len(bad_edges))
	ng = graph.Graph()
	vertex_map = {}
	for vertex in g.vertices:
		vertex_map[vertex] = ng.add_vertex(vertex.point)
	for edge in g.edges:
		if edge not in bad_edges:
			ng.add_edge(vertex_map[edge.src], vertex_map[edge.dst])
	return ng

def apply_conv(session, m, im, size=2048, stride=1024, scale=1, channels=1):
	output = numpy.zeros((im.shape[0], im.shape[1], channels), dtype='float32')
	for x in range(0, im.shape[0] - size, stride) + [im.shape[0] - size]:
		for y in range(0, im.shape[1] - size, stride) + [im.shape[1] - size]:
			conv_input = im[x:x+size, y:y+size, :].astype('float32') / 255.0
			conv_output = session.run(m.outputs, feed_dict={
				m.is_training: False,
				m.inputs: [conv_input],
			})[0, :, :, :]
			startx = size / 2 - stride / 2
			endx = size / 2 + stride / 2
			starty = size / 2 - stride / 2
			endy = size / 2 + stride / 2
			if x == 0:
				startx = 0
			elif x >= im.shape[0] - size - stride:
				endx = size
			if y == 0:
				starty = 0
			elif y >= im.shape[1] - size - stride:
				endy = size
			output[(x+startx)/scale:(x+endx)/scale, (y+starty)/scale:(y+endy)/scale, :] = conv_output[startx/scale:endx/scale, starty/scale:endy/scale, :]
	return output

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description='Run RoadTracer inference.')
	parser.add_argument('modelpath', help='trained model path')
	parser.add_argument('outname', help='output filename to save inferred road network graph')
	parser.add_argument('--s', help='stop threshold (default 0.4)', default=0.4)
	parser.add_argument('--r', help='region (default chicago)', default='chicago')
	parser.add_argument('--t', help='tiles/imagery path')
	parser.add_argument('--g', help='graph path')
	parser.add_argument('--j', help='path to directory containing pytiles.json/starting_locations.json')
	args = parser.parse_args()
	model_path = args.modelpath
	output_fname = args.outname
	BRANCH_THRESHOLD = args.s
	FOLLOW_THRESHOLD = args.s
	REGION = args.r
	if REGION == 'boston':
		TILE_START = geom.Point(1, -1).scale(TILE_SIZE)
	elif REGION == 'chicago':
		TILE_START = geom.Point(-1, -2).scale(TILE_SIZE)
	else:
		TILE_START = geom.Point(-1, -1).scale(TILE_SIZE)
	TILE_END = TILE_START.add(geom.Point(2, 2).scale(TILE_SIZE))

	if args.t: tileloader.tile_dir = args.t
	if args.g: tileloader.graph_dir = args.g
	if args.j:
		tileloader.pytiles_path = os.path.join(args.j, 'pytiles.json')
		tileloader.startlocs_path = os.path.join(args.j, 'starting_locations.json')

	print 'reading tiles'
	tiles = tileloader.Tiles(2, SEGMENT_LENGTH, 16, TILE_MODE)

	print 'initializing model'
	m = model.Model(bn=True, size=2048)
	session = tf.Session()
	m.saver.restore(session, model_path)

	if EXISTING_GRAPH_FNAME is None:
		rect = geom.Rectangle(TILE_START, TILE_END)
		tile_data = tiles.get_tile_data(REGION, rect)

		if USE_TL_LOCATIONS:
			start_loc = random.choice(tile_data['starting_locations'])
		else:
			def match_point(p):
				best_pos = None
				best_distance = None
				for candidate in tile_data['gc'].edge_index.search(p.bounds().add_tol(32)):
					pos = candidate.closest_pos(p)
					distance = pos.point().distance(p)
					if best_pos is None or distance < best_distance:
						best_pos = pos
						best_distance = distance
				return best_pos
			pos1_point = MANUAL_POINT1.add(MANUAL_RELATIVE)
			pos1_pos = match_point(pos1_point)

			if MANUAL_POINT2:
				pos2_point = MANUAL_POINT2.add(MANUAL_RELATIVE)
				pos2_pos = match_point(pos2_point)
			else:
				next_positions = graph.follow_graph(pos1_pos, SEGMENT_LENGTH)
				pos2_point = next_positions[0].point()
				pos2_pos = next_positions[0]

			start_loc = [{
				'point': pos1_point,
				'edge_pos': pos1_pos,
			}, {
				'point': pos2_point,
				'edge_pos': pos2_pos,
			}]

		path = model_utils.Path(tile_data['gc'], tile_data, start_loc=start_loc)
	else:
		g = graph.read_graph(EXISTING_GRAPH_FNAME)
		graph.densify(g, SEGMENT_LENGTH)
		r = g.bounds()
		tile_data = {
			'region': REGION,
			'rect': r.add_tol(WINDOW_SIZE/2),
			'search_rect': r,
			'cache': tiles.cache,
			'starting_locations': [],
		}
		path = model_utils.Path(tiles.get_gc(REGION), tile_data, g=g)

		skip_vertices = set()
		if FILTER_BY_TAG:
			with open(FILTER_BY_TAG, 'r') as f:
				edge_tags = {int(k): v for k, v in json.load(f).items()}
			for edge in g.edges:
				tags = edge_tags[edge.orig_id()]
				if 'highway' not in tags or tags['highway'] in ['pedestrian', 'footway', 'path', 'cycleway', 'construction']:
					for vertex in [edge.src, edge.dst]:
						skip_vertices.add(vertex)
		for vertex in g.vertices:
			vertex.edge_pos = None
			if vertex not in skip_vertices:
				path.prepend_search_vertex(vertex)

	cache_m6 = None
	if M6 and CACHE_M6:
		start_time = time.time()
		big_ims = tile_data['cache'].get(tile_data['region'], tile_data['rect'])
		print 'cache_m6: loaded im in {} sec'.format(time.time() - start_time)
		start_time = time.time()
		cache_m6 = apply_conv(session, m, big_ims['input'], scale=4, channels=64)
		print 'cache_m6: conv in {} sec'.format(time.time() - start_time)

	result = eval([path], m, session, save=SAVE_EXAMPLES, compute_targets=SAVE_EXAMPLES, cache_m6=cache_m6)
	print result
	path.graph.save(output_fname)

'''
import json
ng = graph.Graph()
vertex_map = {}
orig_vertices = set()
for edge in path.graph.edges:
	if not hasattr(edge, 'prob'):
		orig_vertices.add(edge.src)
		orig_vertices.add(edge.dst)
		continue
	for vertex in [edge.src, edge.dst]:
		if vertex not in vertex_map:
			vertex_map[vertex] = ng.add_vertex(vertex.point)
	new_edge = ng.add_edge(vertex_map[edge.src], vertex_map[edge.dst])
	new_edge.prob = edge.prob
ng.save('out.graph')
edge_probs = []
for edge in ng.edges:
	edge_probs.append(int(edge.prob * 100))
with open('out.probs.json', 'w') as f:
	json.dump(edge_probs, f)
interface_vertices = [vertex_map[vertex].id for vertex in orig_vertices if vertex in vertex_map]
with open('out.iface.json', 'w') as f:
	json.dump(interface_vertices, f)
'''

'''
REGION = 'bangkok'
TILE_SIZE = 8192
TILE_START = geom.Point(-3, -3).scale(TILE_SIZE)
TILE_END = TILE_START.add(geom.Point(3, 3).scale(TILE_SIZE))
rect = geom.Rectangle(TILE_START, TILE_END)
tile_data = tiles.get_tile_data(REGION, rect)
for threshold in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
	THRESHOLD_BRANCH = threshold
	THRESHOLD_FOLLOW = threshold
	ng = graph.read_graph('/data/discover-datasets/2018feb06-sea/out_graphs/bangkok_base.graph')
	pg = graph.Graph()
	path = model_utils.Path(None, tile_data, g=pg)
	for edge in ng.edges:
		r = edge.segment().bounds().add_tol(200)
		nearby_edges = path.edge_rtree.intersection((r.start.x, r.start.y, r.end.x, r.end.y))
		if len(list(nearby_edges)) > 0:
			print 'skip {}'.format(edge.id)
			continue
		print 'process {}'.format(edge.id)
		vertex = pg.add_vertex(edge.src.point)
		path.prepend_search_vertex(vertex)
		eval([path], m, session, save=False, compute_targets=False)
	path.graph.save('/data/discover-datasets/2018feb06-sea/out_graphs/bangkok.local256short.{}.graph'.format(threshold))
'''

'''
sat = tiles.cache.get(REGION, tile_data['rect'])['input']
im = numpy.zeros((sat.shape[0]/4, sat.shape[1]/4), dtype='uint8')
for i in xrange(256, sat.shape[0]-256, 256):
	for j in xrange(256, sat.shape[1]-256, 256):
		outputs = session.run(m.outputs, feed_dict={
			m.is_training: False,
			m.inputs: [sat[i:i+256, j:j+256, :].astype('float32') / 255],
		})[0, :, :]
		outputs = outputs.max(axis=2)
		im[i/4:i/4+64, j/4:j/4+64] = (outputs * 255).astype('uint8')

while True:
	for vertex in path.graph.vertices:
		point = vertex.point.sub(tile_data['rect'].start).scale(0.25)
		im[max(point.x-128, 0):min(point.x+128, im.shape[0]), max(point.y-128, 0):min(point.y+128, im.shape[1])] = 0
	if im.max() < 128:
		break
	i, j = numpy.unravel_index(im.argmax(), im.shape)
	point = geom.Point(i, j).scale(4).add(tile_data['rect'].start)
	print 'process {}'.format(point)
	vertex = path.graph.add_vertex(point)
	vertex.edge_pos = None
	path.prepend_search_vertex(vertex)
	eval([path], m, session, save=False, compute_targets=False)
	print 'cleanup'
print 'done'
'''
