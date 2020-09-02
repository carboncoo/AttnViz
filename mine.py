import os
import remi.gui as gui
from remi import start, App

from chart import MultiLayerAttentionMap, HeatMapDataSelectionDialog

class MyApp(App):

    def __init__(self, *args):
        self.res_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'res')
        super(MyApp, self).__init__(*args, static_file_path={'res':self.res_path})

    def main(self):
        self.verticalContainer = gui.Container(width=2000, margin='0px auto', style={'display': 'block', 'overflow': 'hidden'})

        # chart container
        self.chartContainer = gui.Container(width=2000, margin='0px auto', style={'display': 'block', 'overflow': 'hidden'})

        self.selectContainer = gui.Container(width='100%', layout_orientation=gui.Container.LAYOUT_HORIZONTAL, margin='0px', style={'display': 'block', 'overflow': 'auto'})
        self.select_bt = gui.FileUploader('./', width=200, height=30, margin='10px')
        self.select_bt.ondata.do(self.on_data_select)

        self.index_lb = gui.Label('Index: ', width=30, height=30, margin='10px')
        self.index_input = gui.Input(input_type='number', default_value=0, width=40, height=15, margin='10px')
        self.index_input.onchange.do(self.on_index_change)
        self.info_lb = gui.Label('Info: ', width=1000, height=30, margin='10px')

        self.selectContainer.append([self.select_bt, self.index_lb, self.index_input, self.info_lb])
        
        self.chart = MultiLayerAttentionMap(abspath=self.res_path, load_path="/res:", width=2000, height=1000, margin='10px')
        self.chartContainer.append(self.selectContainer)
        self.chartContainer.append(self.chart, "chart")

        self.next_bt = gui.Button('Next', width=200, height=30, margin='10px')
        self.next_bt.onclick.do(self.on_next_button_pressed)
        self.last_bt = gui.Button('Last', width=200, height=30, margin='10px')
        self.last_bt.onclick.do(self.on_last_button_pressed)

        self.verticalContainer.append(self.chartContainer)
        self.verticalContainer.append(self.next_bt)
        self.verticalContainer.append(self.last_bt)
        return self.verticalContainer

    def on_index_change(self, widget, value):
        self.idx.setidx(int(self.index_input.get_value()))
        info = self.chart.update(idx=self.idx)
        self.info_lb.set_text(info)
    
    def on_next_button_pressed(self, widget):
        self.idx = self.chart.get_index()
        self.idx += 1
        info = self.chart.update(idx=self.idx)
        self.index_input.set_value(self.idx)
        self.info_lb.set_text(info)

    def on_last_button_pressed(self, widget):
        self.idx = self.chart.get_index()
        self.idx -= 1
        info = self.chart.update(idx=self.idx)
        self.index_input.set_value(self.idx)
        self.info_lb.set_text(info)

    def on_data_select(self, widget, filedata, filename):
        self.chart.load_binary_data(filedata)
        info = self.chart.update()
        self.idx = self.chart.get_index()
        self.info_lb.set_text(info)

    def data_select_dialog_confirm(self, widget):
        src, tgt, data = self.data_select_dialog.get_filenames()
        self.src_lb.set_text(src)
        self.tgt_lb.set_text(tgt)
        self.data_lb.set_text(data)
        self.chart.load_binary_data(*self.data_select_dialog.get_values())


if __name__ == "__main__":
    start(MyApp, debug=True, address='0.0.0.0', port=8082, start_browser=True, multiple_instance=True)