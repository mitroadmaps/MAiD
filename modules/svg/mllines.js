import _groupBy from 'lodash-es/groupBy';
import _filter from 'lodash-es/filter';
import _flatten from 'lodash-es/flatten';
import _forOwn from 'lodash-es/forOwn';
import _map from 'lodash-es/map';

import { range as d3_range } from 'd3-array';

import {
    geoIdentity as d3_geoIdentity,
    geoPath as d3_geoPath,
    geoStream as d3_geoStream
} from 'd3-geo';

import {
    event as d3_event,
} from 'd3-selection';

import {
	svgOneWaySegments,
	svgPath,
	svgRelationMemberTags,
	svgSegmentWay,
	svgTagClasses
} from './index';

import { osmEntity, osmSimpleMultipolygonOuterMember, osmNode, osmWay } from '../osm';
import { utilDetect } from '../util/detect';


export function svgMLLines(projection, context) {
	var detected = utilDetect();

	function makePathGetter(lines) {
		var cache = {};
		var padding = 5;
		var viewport = projection.clipExtent();
		var paddedExtent = [
			[viewport[0][0] - padding, viewport[0][1] - padding],
			[viewport[1][0] + padding, viewport[1][1] + padding]
		];
		var clip = d3_geoIdentity().clipExtent(paddedExtent).stream;
		var project = projection.stream;
		var path = d3_geoPath()
			.projection({stream: function(output) { return project(clip(output)); }});

		var getPath = function(line) {
			var id = line.id;
			if(line.oid) {
				id = line.oid;
			}
			if (!(id in cache)) {
				var coordinates = [];
				line.nodes.forEach(function(nodeID) {
					coordinates.push(lines[nodeID].loc);
				});
				var geoJSON = {
					'type': 'LineString',
					'coordinates': coordinates,
				};
				cache[id] = path(geoJSON);
			}

			return cache[id];
		};

		getPath.geojson = path;

		return getPath;
	}

	function drawMLLines(selection, lines) {
		var getPath = makePathGetter(lines);

		var ways = [];
		for(var key in lines) {
			var entity = lines[key];
			if(entity.id[0] == 'w') {
				ways.push(entity);
			}
		}
		ways = ways.filter(getPath);
		if(context.hideLines) {
			ways = [];
		}

		var lineSelection = selection.select('.layer-mllines .layer-lines-lines');
		var layergroup = lineSelection.selectAll('g.linegroup')
			.data(['stroke']);

		layergroup = layergroup.enter()
			.append('g')
			.attr('class', function(d) { return 'linegroup line-' + d; })
			.merge(layergroup);

		drawLineGroup = function(selection, cls) {
			var dlines = selection.selectAll('path')
				.data(ways, function(d) { return d.oid; });

			dlines.exit()
				.remove();

			dlines.enter()
				.append('path')
				.attr('class', function(d) { return 'way line mlline ' + cls + ' ' + d.id; })
				.merge(dlines)
				.attr('d', getPath);
		};

		drawLineGroup(lineSelection.select('g.line-stroke'), 'stroke');

		var targetSelection = selection.select('.layer-mllines .layer-lines-targets');
		var targets = [];
		ways.forEach(function(way) {
			for(var i = 0; i < way.nodes.length - 1; i++) {
				var start = lines[way.nodes[i]];
				var end = lines[way.nodes[i + 1]];
				targets.push({
					type: 'Feature',
					id: way.id + '-' + i,
					properties: {
						target: true,
						entity: way,
						index: i
					},
					geometry: {
						type: 'LineString',
						coordinates: [start.loc, end.loc]
					}
				});
			}
		});
		targets = targets.filter(getPath.geojson);

		var tgroup = targetSelection.selectAll('.line.target-allowed')
			.data(targets, function key(d) { return d.id; });
		tgroup.exit()
			.remove();
		tgroup.enter()
			.append('path')
			.on('click', function(d) {
                var mode = context.mode().id;
                if (mode !== 'browse' && mode !== 'select') return;

				var parts = d.id.split('-');
				var id = parts[0] + '-' + parts[1];
				var way = lines[id];
				// add nodes to graph
				insertEntities = function(graph) {
					nodes = [];
					way.nodes.forEach(function(nodeID) {
						var loc = lines[nodeID].loc;
						var exists = function() {
							if(!(nodeID in context.mlInc)) {
								return false;
							}
							var existingNode = graph.entities[context.mlInc[nodeID].id];
							if(!existingNode) {
								return false;
							} else if(existingNode.loc[0] != loc[0] || existingNode.loc[1] != loc[1]) {
								return false;
							}
							return true;
						}();

						if(!exists) {
							var entity = {'loc': loc};
							entity = osmNode(entity);
							graph = graph.replace(entity);
							context.mlInc[nodeID] = entity;
						}
						nodes.push(context.mlInc[nodeID]);
					});
					var entity = {'tags': {'highway': 'residential'}};
					entity = osmWay(entity);
					nodes.forEach(function(node) {
						entity = entity.addNode(node.id);
					});
					graph = graph.replace(entity);
					return graph;
				};
				context.perform(insertEntities);
				delete context.mlLines[id];
				context._map.immediateRedraw();
			})
			.on('contextmenu', function(d) {
				d3_event.preventDefault();
				var parts = d.id.split('-');
				var id = parts[0] + '-' + parts[1];
				context.mlLines = [];
				context._map.immediateRedraw();
			})
			.merge(tgroup)
			.attr('d', getPath.geojson)
			.attr('class', function(d) { return 'way line target target-allowed nocolor ' + d.id; });
	}

	return drawMLLines;
}
