"""
This demo demonstrates how to embed a matplotlib (mpl) plot 
into a PyQt4 GUI application, including:

* Using the navigation toolbar
* Adding data to the plot
* Dynamically modifying the plot's properties
* Processing mpl events
* Saving the plot to a file from a menu

The main goal is to serve as a basis for developing rich PyQt GUI
applications featuring mpl plots (using the mpl OO API).

Eli Bendersky (eliben@gmail.com)
License: this code is in the public domain
Last modified: 19.01.2009
"""
import sys, os, random
from PyQt4.QtCore import *
from PyQt4.QtGui import *

import matplotlib
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure

from process_data import CaponProcessor, Parameter

class ParameterWidget(QWidget):
   
   def __init__(self, param, parent=None):
      super(ParameterWidget, self).__init__(parent)
      
      #print param.name
      
      widgetLayout = QBoxLayout(QBoxLayout.LeftToRight)
      
      self.scrollBar = QScrollBar(Qt.Horizontal)
      self.scrollBar.setFocusPolicy(Qt.StrongFocus)
      self.scrollBar.setMinimum(param.minValue)
      self.scrollBar.setMaximum(param.maxValue)
      self.scrollBar.setValue(param.value)
      self.scrollBar.setMinimumSize(100, 20)
            
      self.connect(self.scrollBar, SIGNAL('valueChanged(int)'), self.scrollBarEventHandler)
      
      self.nameLabel  = QLabel(param.name)
      self.minValueLabel = QLabel(str(param.minValue))
      self.maxValueLabel = QLabel(str(param.maxValue))
      
      self.valueLabel = QLineEdit(str(param.value))
      self.valueLabel.setMaximumSize(100, 20)
      self.connect(self.valueLabel, SIGNAL('editingFinished ()'), self.lineEditEventHandler)
      
      widgetLayout.addStretch(1)
      widgetLayout.addWidget(self.nameLabel)
      widgetLayout.addWidget(self.minValueLabel)
      widgetLayout.addWidget(self.scrollBar)
      widgetLayout.addWidget(self.maxValueLabel)
      widgetLayout.addWidget(self.valueLabel)
      
      self.setLayout(widgetLayout)
      
      self.param = param
      self.lock = False
      
   def scrollBarEventHandler(self):
      if self.lock is False:
         self.lock = True
         v = self.scrollBar.value()
         self.param.updateValue(v)
         self.valueLabel.setText(str(self.param.value))
         self.reProcessAndDraw()
         self.lock = False
      
   def lineEditEventHandler(self):
      if self.lock is False:
         self.lock = True
         try:
            v = int(self.valueLabel.text())
         
            self.param.updateValue(v)
            if self.param.isUpdated():
               self.scrollBar.setValue(v)
               self.reProcessAndDraw()
         except:
            print "Unexpected error:", sys.exc_info()[0]
         self.lock = False
      
   def reProcessAndDraw(self):
      if self.param.needReProcessing:
         self.parent().parent().parent().reProcess = True # not so good
      else:
         self.parent().parent().parent().reProcess = False
      self.parent().parent().parent().on_draw()
      

class SlidersGroup(QGroupBox):

   valueChanged = pyqtSignal(int)

   def __init__(self, title, parameters, parent=None):
      super(SlidersGroup, self).__init__(title, parent)
     
      slidersLayout = QBoxLayout(QBoxLayout.TopToBottom)
      
      for param in parameters.__dict__.itervalues():
         
         if param.__class__.__name__ == 'Parameter':
            
            slidersLayout.addWidget(ParameterWidget(param, self))
         
      self.setLayout(slidersLayout)    
     
   def setValue(self, value):    
      self.slider.setValue(value)    

   def setMinimum(self, value):    
      self.scrollBar.setMinimum(value)    

   def setMaximum(self, value):    
      self.scrollBar.setMaximum(value)   


class AppForm(QMainWindow):
   def __init__(self, parent=None):
      QMainWindow.__init__(self, parent)
      self.setWindowTitle('Demo: Capon on Speed')
      
      self.datatag = []
      self.datatag.append('motion phantom 16x 3.4 MHz 20dB30dB')
      self.multiFile = True
      
      self.create_menu()
      self.create_status_bar()   
      
      self.idx_of_cur_file = 0
      self.load_data(self.idx_of_cur_file)
      
      
   def load_data(self, index):
      self.idx_of_cur_file = index
      self.caponProcessor = CaponProcessor(self.datatag[index], self.multiFile)
      self.reProcess = True
      self.create_main_frame()
      self.on_draw()


   def save_plot(self):
      file_choices = "PNG (*.png)|*.png"
      
      path = unicode(QFileDialog.getSaveFileName(self,
                      'Save file', '',
                      file_choices))
      if path:
         self.canvas.print_figure(path, dpi=self.dpi)
         self.statusBar().showMessage('Saved to %s' % path, 2000)
         
   def dump_frames_and_video(self):
      
      tmp_reprocess = self.reProcess
      self.reProcess = True
      capon = self.caponProcessor
      
      if not os.path.exists('video'):
         os.mkdir('video')
         
      os.chdir('video')
      
      files = []
      for i in range(capon.image_idx.minValue, capon.image_idx.maxValue+1):

         capon.image_idx.value = i

         self.on_draw()

         if capon.save_single_plots.value is 1:
            # Save just the portion _inside_ the  axis's boundaries
            extent1 = self.axes.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())  
            extent2 = self.axes2.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
            
            self.fig.savefig('_tmp%03d_1.png'%i, bbox_inches=extent1.expanded(1.4, 1.25))
            self.fig.savefig('_tmp%03d_2.png'%i, bbox_inches=extent2.expanded(1.4, 1.25))
            
         else:
            fname = '_tmp%03d.png'%i
            print 'Saving frame', fname
            self.fig.savefig(fname)
            files.append(fname)
         
      self.reProcess = tmp_reprocess
      self.caponProcessor.image_idx.value = 1
      os.chdir('..')
      
      self.make_video()
         
   def make_video(self):
      print 'Rendering video'
      if os.path.exists('video'):
         os.chdir('video')
      
         capon = self.caponProcessor
      
         fps = capon.fps.value
      
         if capon.save_single_plots.value is 1:
            video_file_name = '%s_M=%d_L=%d_K=%d_d=%f_plot1.avi'%(self.datatag[self.idx_of_cur_file].replace(' ', '_'), capon.Nm, capon.L.value, capon.K.value, 10.0**capon.d.value)
            cmd = "mencoder mf://_tmp*_1.png -mf fps=%d -ovc lavc -lavcopts vcodec=mjpeg -o %s"%(fps,video_file_name)
            os.system(cmd)
            
            video_file_name = '%s_M=%d_L=%d_K=%d_d=%f_plot2.avi'%(self.datatag[self.idx_of_cur_file].replace(' ', '_'), capon.Nm, capon.L.value, capon.K.value, 10.0**capon.d.value)
            cmd = "mencoder mf://_tmp*_2.png -mf fps=%d -ovc lavc -lavcopts vcodec=mjpeg -o %s"%(fps,video_file_name)
            os.system(cmd)
         else:
            video_file_name = '%s_M=%d_L=%d_K=%d_d=%f.avi'%(self.datatag[self.idx_of_cur_file].replace(' ', '_'), capon.Nm, capon.L.value, capon.K.value, 10.0**capon.d.value)

            print 'Making movie animation.mpg - this make take a while'
            #cmd = "mencoder mf://_tmp*.png -mf type=png:fps=%d -ovc lavc -lavcopts vcodec=wmv2 -oac copy -o %s"%(fps,video_file_name)
            #cmd = "mencoder mf://*.png -mf fps=10:type=jpg -ovc lavc -lavcopts vcodec=mpeg4:mbd=2:trell -oac copy -o output.avi"
            #cmd = "mencoder mf://_tmp*.png -mf type=png:fps=%d -ovc copy -oac copy -o %s"%(fps,video_file_name)
            cmd = "mencoder mf://_tmp*.png -mf fps=%d -ovc lavc -lavcopts vcodec=mjpeg -o %s"%(fps,video_file_name)
            #print cmd
            os.system(cmd)
      else:
         print 'No frames found in video folder'
      
      #for file in files:
      #   os.remove(file)
      
      os.chdir('..')
      
   def save_figs(self):
      from datetime import datetime
      d = datetime.time(datetime.now())
      
      extent1 = self.axes.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())  
      extent2 = self.axes2.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
      extent3 = self.axes3.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
      extent4 = self.axes4.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
            
      self.fig.savefig('DAS_%d%d%d%d'%(d.hour, d.minute, d.second, d.microsecond),          bbox_inches=extent1.expanded(1.4, 1.25))
      self.fig.savefig('Capon_%d%d%d%d'%(d.hour, d.minute, d.second, d.microsecond),        bbox_inches=extent2.expanded(1.4, 1.25))
      self.fig.savefig('BeamPattern_%d%d%d%d'%(d.hour, d.minute, d.second, d.microsecond),  bbox_inches=extent3.expanded(1.4, 1.25))
      self.fig.savefig('Profiles_%d%d%d%d'%(d.hour, d.minute, d.second, d.microsecond),     bbox_inches=extent4.expanded(1.4, 1.25))
      
   def on_about(self):
      msg = """ A demo of using PyQt with matplotlib:
      
       * Use the matplotlib navigation bar
       * Add values to the text box and press Enter (or click "Draw")
       * Show or hide the grid
       * Drag the slider to modify the width of the bars
       * Save the plot to a file using the File menu
       * Click on a bar to receive an informative message
      """
      QMessageBox.about(self, "About the demo", msg.strip())
      
   def on_pick(self, event):
      # The event received here is of the type
      # matplotlib.backend_bases.PickEvent
      #
      # It carries lots of information, of which we're using
      # only a small amount here.
      # 

      #msg = "You've clicked on %s with coords:\n (%d, %d)" % (event.button, event.xdata, event.ydata)
      #QMessageBox.information(self, "Click!", msg)
      
      self.caponProcessor.profilePos[0] = event.xdata
      self.caponProcessor.profilePos[1] = event.ydata
      
      print "Coordinates: (%d, %d)", (event.xdata, event.ydata) 
      
      self.reProcess = False
      self.on_draw()
   
   def on_draw(self):
      """ Redraws the figure
      """
      # clear the axes and redraw the plots
      self.axes.clear() 
      self.axes2.clear()
      self.axes3.clear()
      self.axes4.clear()       
      #self.axes.grid(self.grid_cb.isChecked())
      
      if self.reProcess:
         self.caponProcessor.processData()
      self.caponProcessor.plot(self.axes, self.axes2, self.axes3, self.axes4)
      
      self.canvas.draw() # TODO: this is slow, draw only changed components
   
   def create_main_frame(self):
      self.main_frame = QWidget()
      
      # Create the mpl Figure and FigCanvas objects. 
      # 5x4 inches, 100 dots-per-inch
      #
      self.dpi = 100
      #self.fig = Figure((20.0, 15.0), dpi=self.dpi) # 1080*600
      self.fig = Figure((6.0, 12.0), dpi=self.dpi)
      self.canvas = FigureCanvas(self.fig)
      self.canvas.setParent(self.main_frame)
      
      # Since we have only one plot, we can use add_axes 
      # instead of add_subplot, but then the subplot
      # configuration tool in the navigation toolbar wouldn't
      # work.
      #
      import matplotlib.gridspec as gridspec
      grid = (11,13)
      halfGridX = grid[0]/2
      halfGridY = grid[1]/2
      gs = gridspec.GridSpec(grid[0], grid[1])
      
      
      self.axes  = self.fig.add_subplot(gs[:halfGridX+1,:halfGridY], aspect='1')
      self.axes2 = self.fig.add_subplot(gs[:halfGridX+1,halfGridY+1:], aspect='1')
      self.axes3 = self.fig.add_subplot(gs[halfGridX+2:,:halfGridY])#, aspect='0.4')
      self.axes4 = self.fig.add_subplot(gs[halfGridX+2:,halfGridY+1:])#, aspect='0.4')
      
      #self.axes  = self.fig.add_subplot(221, aspect='1')
      #self.axes2 = self.fig.add_subplot(222, aspect='1')
      #self.axes3 = self.fig.add_subplot(223)
      #self.axes4 = self.fig.add_subplot(224)
      
      #self.axes.invert_yaxis()
      #self.axes2.invert_yaxis()
      
      #self.fig.tight_layout(pad=2, h_pad=2, w_pad=2)
      
      # Bind the 'pick' event for clicking on one of the bars
      #
      self.canvas.mpl_connect('button_press_event', self.on_pick)
      
      # Create the navigation toolbar, tied to the canvas
      #
      self.mpl_toolbar = NavigationToolbar(self.canvas, self.main_frame)
      
      # Other GUI controls
      self.video_button = QPushButton("&Dump frames and video")
      self.connect(self.video_button, SIGNAL('clicked()'), self.dump_frames_and_video)
      
      self.video_button2 = QPushButton("&Make video")
      self.connect(self.video_button2, SIGNAL('clicked()'), self.make_video)
      
      self.save_figs_button = QPushButton("&Save Individual Figures")
      self.connect(self.save_figs_button, SIGNAL('clicked()'), self.save_figs)
      
      self.file_list = QComboBox()
      self.file_list.addItems(self.datatag)
      self.file_list.setCurrentIndex(self.idx_of_cur_file)
      self.connect(self.file_list, SIGNAL('currentIndexChanged(int)'), self.load_data)
#      
      self.parameterSliders = SlidersGroup('Parameters', self.caponProcessor, self)
      #
      # Layout with box sizers
      # 
      vbox = QVBoxLayout()
      
      for w in [ self.parameterSliders, self.video_button, self.video_button2, self.file_list, self.save_figs_button]:
         vbox.addWidget(w)
         vbox.setAlignment(w, Qt.AlignRight)
      
      hbox = QHBoxLayout()
      vbox2 = QVBoxLayout()
      vbox2.addWidget(self.canvas)
      vbox2.addWidget(self.mpl_toolbar)
      
      hbox.addLayout(vbox2)
      hbox.addLayout(vbox)
      
      self.main_frame.setLayout(hbox)
      self.setCentralWidget(self.main_frame)
   
   def create_status_bar(self):
      self.status_text = QLabel("This is a demo of capon beamforming")
      self.statusBar().addWidget(self.status_text, 1)
       
   def create_menu(self):        
      self.file_menu = self.menuBar().addMenu("&File")
      
      load_file_action = self.create_action("&Save plot",
         shortcut="Ctrl+S", slot=self.save_plot,
         tip="Save the plot")
      quit_action = self.create_action("&Quit", slot=self.close,
         shortcut="Ctrl+Q", tip="Close the application")
      
      self.add_actions(self.file_menu,
         (load_file_action, None, quit_action))
      
      self.help_menu = self.menuBar().addMenu("&Help")
      about_action = self.create_action("&About",
         shortcut='F1', slot=self.on_about,
         tip='About the demo')
      
      self.add_actions(self.help_menu, (about_action,))
   
   def add_actions(self, target, actions):
      for action in actions:
         if action is None:
            target.addSeparator()
         else:
            target.addAction(action)
   
   def create_action(self, text, slot=None, shortcut=None,
                     icon=None, tip=None, checkable=False,
                     signal="triggered()"):
      action = QAction(text, self)
      if icon is not None:
         action.setIcon(QIcon(":/%s.png" % icon))
      if shortcut is not None:
         action.setShortcut(shortcut)
      if tip is not None:
         action.setToolTip(tip)
         action.setStatusTip(tip)
      if slot is not None:
         self.connect(action, SIGNAL(signal), slot)
      if checkable:
         action.setCheckable(True)
      return action


def main():
   app = QApplication(sys.argv)
   form = AppForm()
   form.show()
   app.exec_()


if __name__ == "__main__":
   main()
