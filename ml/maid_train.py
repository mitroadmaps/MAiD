import sys
sys.path.append('../lib')

from discoverlib import geom, graph
import maid_model as model
import tileloader

from collections import deque
import numpy
import math
import os
import os.path
from PIL import Image
import random
import scipy.ndimage
import sys
import tensorflow as tf
import time

import argparse
parser = argparse.ArgumentParser(description='Train a MAiD model.')
parser.add_argument('modelpath', help='path to save model')
parser.add_argument('--t', help='tiles/imagery path')
parser.add_argument('--g', help='graph path')
parser.add_argument('--a', help='angles path')
parser.add_argument('--j', help='path to directory containing pytiles.json/starting_locations.json')
args = parser.parse_args()

if args.a:
	tileloader.angles_dir = args.a
else:
	print 'error: --a option (angles path) not specified'
	sys.exit(1)

if args.t: tileloader.tile_dir = args.t
if args.g: tileloader.graph_dir = args.g
if args.j:
	tileloader.pytiles_path = os.path.join(args.j, 'pytiles.json')
	tileloader.startlocs_path = os.path.join(args.j, 'starting_locations.json')

MODEL_BASE = args.modelpath
WINDOW_SIZE = 512
NUM_TRAIN_TILES = 1024
TILE_SIZE = 4096
RECT_OVERRIDE = None
NUM_BUCKETS = 64
MASK_NEAR_ROADS = False

tiles = tileloader.Tiles(2, 20, NUM_TRAIN_TILES+8, 'sat')
tiles.prepare_training()
train_tiles = tiles.train_tiles

# initialize model and session
print 'initializing model'
m = model.Model(bn=True, size=512)
session = tf.Session()
model_path = MODEL_BASE + '/model_latest/model'
best_path = MODEL_BASE + '/model_best/model'
if os.path.isfile(model_path + '.meta'):
	print '... loading existing model'
	m.saver.restore(session, model_path)
else:
	print '... initializing a new model'
	session.run(m.init_op)

def get_tile_rect(tile):
	if RECT_OVERRIDE:
		return RECT_OVERRIDE
	p = geom.Point(tile.x, tile.y)
	return geom.Rectangle(
		p.scale(TILE_SIZE),
		p.add(geom.Point(1, 1)).scale(TILE_SIZE)
	)

def get_tile_example(tile, tries=10):
	rect = get_tile_rect(tile)

	# pick origin: must be multiple of the output scale
	origin = geom.Point(random.randint(0, rect.lengths().x/4 - WINDOW_SIZE/4), random.randint(0, rect.lengths().y/4 - WINDOW_SIZE/4))
	origin = origin.scale(4)
	origin = origin.add(rect.start)

	tile_origin = origin.sub(rect.start)
	big_ims = tiles.cache.get_window(tile.region, rect, geom.Rectangle(tile_origin, tile_origin.add(geom.Point(WINDOW_SIZE, WINDOW_SIZE))))
	input = big_ims['input'].astype('float32') / 255.0
	target = big_ims['angles'].astype('float32') / 255.0
	if numpy.count_nonzero(target.max(axis=2)) < 64 and tries > 0:
		return get_tile_example(tile, tries - 1)
	example = {
		'region': tile.region,
		'origin': origin,
		'input': input,
		'target': target,
	}
	if MASK_NEAR_ROADS:
		mask = target.max(axis=2) > 0
		mask = scipy.ndimage.morphology.binary_dilation(mask, iterations=9)
		example['mask'] = mask
	return example

def get_example(traintest='train'):
	if traintest == 'train':
		tile = random.choice(train_tiles)
	elif traintest == 'test':
		tile = random.choice([tile for tile in tiles.all_tiles if tile.region == 'chicago' and tile.x >= -1 and tile.x < 1 and tile.y >= -2 and tile.y < 0])
	return get_tile_example(tile)

val_examples = [get_example('test') for _ in xrange(2048)]

def vis_example(example, outputs=None):
	x = numpy.zeros((WINDOW_SIZE, WINDOW_SIZE, 3), dtype='uint8')
	x[:, :, :] = example['input'] * 255
	x[WINDOW_SIZE/2-2:WINDOW_SIZE/2+2, WINDOW_SIZE/2-2:WINDOW_SIZE/2+2, :] = 255

	gc = tiles.get_gc(example['region'])
	rect = geom.Rectangle(example['origin'], example['origin'].add(geom.Point(WINDOW_SIZE, WINDOW_SIZE)))
	for edge in gc.edge_index.search(rect):
		start = edge.src.point
		end = edge.dst.point
		for p in geom.draw_line(start.sub(example['origin']), end.sub(example['origin']), geom.Point(WINDOW_SIZE, WINDOW_SIZE)):
			x[p.x, p.y, 0:2] = 0
			x[p.x, p.y, 2] = 255

	for i in xrange(WINDOW_SIZE):
		for j in xrange(WINDOW_SIZE):
			di = i - WINDOW_SIZE/2
			dj = j - WINDOW_SIZE/2
			d = math.sqrt(di * di + dj * dj)
			a = int((math.atan2(dj, di) - math.atan2(0, 1) + math.pi) * NUM_BUCKETS / 2 / math.pi)
			if a >= NUM_BUCKETS:
				a = NUM_BUCKETS - 1
			elif a < 0:
				a = 0
			elif d > 100 and d <= 120 and example['target'] is not None:
				x[i, j, 0] = example['target'][WINDOW_SIZE/8, WINDOW_SIZE/8, a] * 255
				x[i, j, 1] = example['target'][WINDOW_SIZE/8, WINDOW_SIZE/8, a] * 255
				x[i, j, 2] = 0
			elif d > 70 and d <= 90 and outputs is not None:
				x[i, j, 0] = outputs[WINDOW_SIZE/8, WINDOW_SIZE/8, a] * 255
				x[i, j, 1] = outputs[WINDOW_SIZE/8, WINDOW_SIZE/8, a] * 255
				x[i, j, 2] = 0
	return x

'''
for i in xrange(128):
	im = vis_example(val_examples[i])
	Image.fromarray(im).save('/home/ubuntu/data/{}.png'.format(i))
'''

best_loss = None

def epoch_to_learning_rate(epoch):
	if epoch < 20:
		return 1e-3
	elif epoch < 40:
		return 1e-4
	elif epoch < 60:
		return 1e-5
	else:
		return 1e-6

for epoch in xrange(80):
	start_time = time.time()
	train_losses = []
	for _ in xrange(1024):
		examples = [get_example('train') for _ in xrange(model.BATCH_SIZE)]
		feed_dict = {
			m.is_training: True,
			m.inputs: [example['input'] for example in examples],
			m.targets: [example['target'] for example in examples],
			m.learning_rate: epoch_to_learning_rate(epoch),
		}
		if MASK_NEAR_ROADS:
			feed_dict[m.mask] = [example['mask'] for example in examples]
		_, loss = session.run([m.optimizer, m.loss], feed_dict=feed_dict)
		train_losses.append(loss)

	train_loss = numpy.mean(train_losses)
	train_time = time.time()

	val_losses = []
	for i in xrange(0, len(val_examples), model.BATCH_SIZE):
		examples = val_examples[i:i+model.BATCH_SIZE]
		feed_dict = {
			m.is_training: False,
			m.inputs: [example['input'] for example in examples],
			m.targets: [example['target'] for example in examples],
		}
		if MASK_NEAR_ROADS:
			feed_dict[m.mask] = [example['mask'] for example in examples]
		loss = session.run([m.loss], feed_dict=feed_dict)
		val_losses.append(loss)

	val_loss = numpy.mean(val_losses)
	val_time = time.time()

	#outputs = session.run(m.angle_outputs, feed_dict={
	#	m.is_training: False,
	#	m.inputs: [example[1] for example in val_examples[:model.BATCH_SIZE]],
	#})
	#for i in xrange(model.BATCH_SIZE):
	#	im = vis_example(val_examples[i], outputs=outputs[i, :])
	#	Image.fromarray(im).save('/home/ubuntu/data/{}_{}.png'.format(epoch, i))

	print 'iteration {}: train_time={}, val_time={}, train_loss={}, val_loss={}/{}'.format(epoch, int(train_time - start_time), int(val_time - train_time), train_loss, val_loss, best_loss)

	m.saver.save(session, model_path)
	if best_loss is None or val_loss < best_loss:
		best_loss = val_loss
		m.saver.save(session, best_path)

'''
outputs = session.run(m.angle_outputs, feed_dict={
	m.is_training: False,
	m.inputs: [val_examples[0][1]],
})
'''

'''
for i in xrange(0, len(val_examples[0:256]), model.BATCH_SIZE):
	examples = val_examples[i:i+model.BATCH_SIZE]
	outputs = session.run(m.angle_outputs, feed_dict={
		m.is_training: False,
		m.inputs: [example[1] for example in examples],
	})
	for j in xrange(model.BATCH_SIZE):
		im = vis_example(examples[j], outputs=outputs[j, :])
		Image.fromarray(im).save('/home/ubuntu/data/{}.png'.format(i+j))
'''
