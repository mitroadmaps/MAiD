import {
    event as d3_event,
    select as d3_select
} from 'd3-selection';

import { d3keybinding as d3_keybinding } from '../lib/d3.keybinding.js';

import { t, textDirection } from '../util/locale';
import { svgIcon } from '../svg';
import { uiCmd } from './cmd';
import { uiTooltipHtml } from './tooltipHtml';
import { tooltip } from '../util/tooltip';

export function uiML(context) {
    function ml() {
        d3_event.preventDefault();
    	var bounds = context.extent();
		var xhr = new XMLHttpRequest();
		xhr.open('GET', '/ml?x1=' + bounds[0][0] + '&y1=' + bounds[0][1] + '&x2=' + bounds[1][0] + '&y2=' + bounds[1][1], true);
		xhr.onreadystatechange = function() {
			if(xhr.readyState !== 4) {
				return;
			} else if(xhr.status !== 200) {
				console.log('ml: bad status ' + xhr.status);
				return;
			}
			var response = JSON.parse(xhr.responseText);
			context.mlLines = {};
			response['ml_full'].forEach(function(entity) {
				context.mlLines[entity.id] = entity;
			});
			context.jumps = response['jumps'];
			context._map.immediateRedraw();
		};
		xhr.send(null);
    }

    function jump() {
		if(context.jumps.length == 0) {
			return;
		}
		var nextRect = context.jumps[0];
		context.jumps = context.jumps.slice(1);
		context.extent(nextRect);
    }

    function hide() {
		if(d3_event.key != 'h') {
			return;
		}
		if(d3_event.type == 'keyup') {
			context.hideLines = false;
		} else if(d3_event.type == 'keydown') {
			context.hideLines = true;
		}
		context._map.immediateRedraw();
    }

    return function(selection) {
        var button = selection.append('button')
            .attr('class', 'col6')
            .attr('tabindex', -1)
            .on('click', ml)
            .style('background', '#fff');

        button
            .append('span')
            .attr('class', 'label')
            .text('Run ML');

        var button = selection.append('button')
            .attr('class', 'col6')
            .attr('tabindex', -1)
            .on('click', jump)
            .style('background', '#fff');

        button
            .append('span')
            .attr('class', 'label')
            .text('Jump');

        var keybinding = d3_keybinding('jump')
            .on(t('jump.key'), jump);
        d3_select(document)
            .call(keybinding);

        var keybinding = d3_keybinding('hide')
        	.on('keydown', hide)
        	.on('keyup', hide);
    	d3_select('body')
    		.on('keydown', hide)
    		.on('keyup', hide);
    };
}
