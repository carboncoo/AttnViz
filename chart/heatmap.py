import os
import time
import numpy as np
import remi.gui as gui
from remi import start, App
from tempfile import TemporaryFile

from pyecharts import options as opts
from pyecharts.charts import HeatMap
from pyecharts.faker import Faker
from pyecharts.commons.utils import JsCode

import torch

from .data import MultiAttentionMeanDataGenerator, sort_keys

class HeatMapWidget(gui.Container):

    def __init__(self, abspath="/res", load_path="/res:", chart_args={'width':2000, 'height':800, 'margin': '10px'}, *args, **kwargs):
        super(HeatMapWidget, self).__init__(*args, **kwargs)
        self.abspath = abspath
        self.current_filename = str(time.time())
        self.load_path = load_path
        self.chart_args = chart_args
        self.chart = self.get_chart()

        self.append(self.chart, "chart")

        self._data = None
        self.idx = -1
    
    def get_index(self):
        return self._data.idx

    def get_chart(self):

        chart = gui.Widget( _type='iframe', **self.chart_args)
        chart.attributes['width'] = '100%'
        chart.attributes['height'] = '100%'
        chart.attributes['controls'] = 'true'
        chart.style['border'] = 'none'
        return chart

    def render(self, x_lb, y_lb, value):
        info = value['info'] if 'info' in value else ""
        c = (
            HeatMap(init_opts=opts.InitOpts(width='{}px'.format(max(50*len(x_lb), 1000)), height='{}px'.format(max(20*len(y_lb), 500))))
            .add_xaxis(x_lb)
            .set_global_opts(
                legend_opts=opts.LegendOpts(type_='scroll'),
                visualmap_opts=opts.VisualMapOpts(is_show=True, 
                                                orient='horizontal', 
                                                pos_left='center', 
                                                pos_bottom='-20',
                                                range_color=['#ffffff', '#000000']),
                tooltip_opts=opts.TooltipOpts(is_show=True, formatter=JsCode("""function(params){
                            return params.data['name'] + ' : ' + (params.data['value'][2] / 100).toFixed(2) 
                        }
                        """)),
                xaxis_opts=opts.AxisOpts(axislabel_opts={'rotate':45, 'interval': 0}, interval=0, splitline_opts=opts.SplitLineOpts(is_show=True, linestyle_opts={'color':'black', 'width':1})),
                yaxis_opts=opts.AxisOpts(axislabel_opts={'interval': 0}, is_inverse=True, interval=0,
                                        splitline_opts=opts.SplitLineOpts(is_show=True, linestyle_opts={'color':'black', 'width':1}))
            )
        )
        keys = sort_keys(value['weights'].keys())
        for k in keys:
            v = value['weights'][k]
            c.add_yaxis(
                k,
                yaxis_data=y_lb,
                value=v,
                is_selected=('ref' in k),
                itemstyle_opts=opts.ItemStyleOpts(opacity=0.8)
            )
        old_filename = os.path.join(self.abspath, self.current_filename)
        if os.path.exists(old_filename):
            os.remove(old_filename)
        self.current_filename = str(time.time()) + '.html'
        c.render(os.path.join(self.abspath, self.current_filename))
        self.remove_child('chart')
        self.chart = self.get_chart()
        self.chart.attributes['src'] = self.load_path + self.current_filename
        self.append(self.chart, "chart")
        return info

    def fake_data(self):
        import random
        x_lb = Faker.clock
        y_lb = Faker.week
        value = [[i, j, random.randint(0, 50)] for i in range(24) for j in range(7)]
        return (x_lb, y_lb, {'score': value})

    def load_data(self, x_lbs, y_lbs, values):
        data = list(zip(x_lbs, y_lbs, values))
        data = sorted(data, key=lambda x: len(x[0]) + len(x[1]))
        self._data = DataGenerator(data)
        self.update()

    def update(self, idx=None, forward=True):
        if self._data is None:
            return self.render(*self.fake_data())
        if idx is not None:
            return self.render(*self._data[idx])
        if forward:
            return self.render(*self._data.next())
        else:
            return self.render(*self._data.last())

    def reorder(self, expr):
        success = self._data.sorted_by(expr)
        if success:
            return self.update(idx=0)
        else:
            return None

class MultiLayerAttentionMap(HeatMapWidget):

    def __init__(self, *args, **kwargs):
        super(MultiLayerAttentionMap, self).__init__(*args, **kwargs)
    
    def load_binary_data(self, v):
        open('tmp.pt','wb').write(v)
        all_data = torch.load('tmp.pt', map_location=torch.device('cpu'))
        all_data = sorted(all_data, key=lambda x: len(x['src']) + len(x['tgt']))
        self._data = MultiAttentionMeanDataGenerator(all_data)

class HeatMapDataSelectionDialog(gui.GenericDialog):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.src_input = gui.FileUploader('./', width=200, height=30, margin='10px')
        self.tgt_input = gui.FileUploader('./', width=200, height=30, margin='10px')
        self.data_input = gui.FileUploader('./', width=200, height=30, margin='10px')
        self.src_input.ondata.do(self.src_input_ondata)
        self.tgt_input.ondata.do(self.tgt_input_ondata)
        self.data_input.ondata.do(self.data_input_ondata)
        self.add_field_with_label('src_input', 'Source File', self.src_input)
        self.add_field_with_label('tgt_input', 'Target File', self.tgt_input)
        self.add_field_with_label('data_input', 'Data File', self.data_input)
    
    def src_input_ondata(self, widget, filedata, filename):
        self._src = filedata
        self._src_name = filename
    
    def tgt_input_ondata(self, widget, filedata, filename):
        self._tgt = filedata
        self._tgt_name = filename
    
    def data_input_ondata(self, widget, filedata, filename):
        self._data = filedata
        self._data_name = filename

    def get_filenames(self):
        return self._src_name, self._tgt_name, self._data_name
    
    def get_values(self):
        return self._src, self._tgt, self._data
