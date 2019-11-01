
import wx
from wx.lib.pubsub import pub
import csv
import shutil
from tempfile import NamedTemporaryFile
import numpy as np
import random
import cv2
import os

########################################################################
class ViewerPanel(wx.Panel):
    """"""

    #----------------------------------------------------------------------
    def __init__(self, parent):
        """Constructor"""
        random.seed(222)

        wx.Panel.__init__(self, parent)
        
        width, height = wx.DisplaySize()

        self.class_labels = {0: 'background', 16: 'bird', 17: 'buddy', 18: 'jade', 91: 'buddy', 92: 'jade'}
        self.class_labels_rev = {'background': 0, 'jade': 92, 'buddy': 91}

        self.csvFields = ['frame','width','height','class_id','xmin','ymin','xmax','ymax']

        self.startingRow = 1
#        self.startingRow = 1
        self.rowNumber = 0

        self.stratified = True
        self.csvRows = []
        self.percentVal = 0.25

        # 20180706_215_cats_807,138,231,94,194,1
        self.frame = ''
        self.xmin = 0
        self.xmax = 0
        self.ymin = 0
        self.ymax = 0
        self.classification = 0
        self.newClassification = 0

        self.currentPicture = 0
        self.totalPictures = 0
        #self.photoMaxSize = 300
        self.photoMaxSize = height - 200

        pub.subscribe(self.loadTrainingCSV, ("update images"))

        self.slideTimer = wx.Timer(None)
        self.slideTimer.Bind(wx.EVT_TIMER, self.update)

        self.cropOnNext = True

        self.layout()

    #----------------------------------------------------------------------
    def layout(self):
        """
        Layout the widgets on the panel
        """

        self.mainSizer = wx.BoxSizer(wx.VERTICAL)
        btnSizer = wx.BoxSizer(wx.HORIZONTAL)

        self.img = wx.Image(self.photoMaxSize,self.photoMaxSize)
        self.imageCtrl = wx.StaticBitmap(self, wx.ID_ANY,
                                         wx.Bitmap(self.img))
        self.mainSizer.Add(self.imageCtrl, 0, wx.ALL|wx.CENTER, 5)
        self.imageLabel = wx.StaticText(self, label="")
        self.mainSizer.Add(self.imageLabel, 0, wx.ALL|wx.CENTER, 5)

        btnData = [("Previous", btnSizer, self.onPrevious),
                   ("Next", btnSizer, self.onNext),
                   ("Crop", btnSizer, self.onCrop),
                   ("Update Class", btnSizer, self.onUpdateCSV),
                   ("Save", btnSizer, self.onDone),
                   ("Validate", btnSizer, self.onValidate)]
        for data in btnData:
            label, sizer, handler = data
            self.btnBuilder(label, sizer, handler)

        self.mainSizer.Add(btnSizer, 0, wx.CENTER)
        self.SetSizer(self.mainSizer)

    #----------------------------------------------------------------------
    def btnBuilder(self, label, sizer, handler):
        """
        Builds a button, binds it to an event handler and adds it to a sizer
        """
        btn = wx.Button(self, label=label)
        btn.Bind(wx.EVT_BUTTON, handler)
        sizer.Add(btn, 0, wx.ALL|wx.CENTER, 5)

    #----------------------------------------------------------------------
    def loadImage(self, image):
        """"""
        self.img = wx.Image(image, wx.BITMAP_TYPE_ANY)

        # scale the image, preserving the aspect ratio
        W = self.img.GetWidth()
        H = self.img.GetHeight()
        if W > H:
            NewW = self.photoMaxSize
            NewH = self.photoMaxSize * H / W
        else:
            NewH = self.photoMaxSize
            NewW = self.photoMaxSize * W / H
        img = self.img.Scale(NewW,NewH)

        xscale = img.GetWidth()/W
        yscale = img.GetHeight()/H

        sxmin = int(xscale * self.xmin)
        symin = int(yscale * self.ymin)
        sxmax = int(xscale * self.xmax)
        symax = int(yscale * self.ymax)

        imgBit = wx.BitmapFromImage(img)
        dc = wx.MemoryDC(imgBit)
        dc.SetPen(wx.Pen(wx.RED, 1))
        text = self.class_labels[self.classification]
        tw, th = dc.GetTextExtent(text)
        dc.SetTextForeground(wx.RED)
        dc.DrawText(text, sxmax-((sxmax-sxmin)/2)-tw, symax-th)
        dc.SetBrush(wx.Brush(wx.RED, wx.TRANSPARENT)) #set brush transparent for non-filled rectangle
        dc.DrawRectangle(sxmin,symin,sxmax-sxmin,symax-symin)
        dc.SelectObject(wx.NullBitmap)

        self.imageCtrl.SetBitmap(imgBit)
        label = "Row:{:04d}   Image: {}{:03d}x{:03d}   xmin:{:03d}   xmax:{:03d}   ymin:{:03d}   ymax:{:03d}   class: {}".format(self.rowNumber+1, self.frame, W,H,self.xmin, self.xmax, self.ymin, self.ymax, self.class_labels[self.classification])
        self.imageLabel.SetLabel(label)
        self.Refresh()
        pub.sendMessage(topicName="resize", msg="")

    #----------------------------------------------------------------------
    def cropPicture(self):
        image = cv2.imread(self.imagePath + '/' + self.frame)
#        image = img_to_array(image)
        crop = image[self.ymin:self.ymax,self.xmin:self.xmax].copy()
        crop = cv2.resize(crop, (100, 100))
        dir = self.imagePath + '/crops'
        if not os.path.isdir(dir):
            os.mkdir(dir)
        dir2 = dir + '/' + self.class_labels[self.classification]
        if not os.path.isdir(dir2):
            os.mkdir(dir2)
        filename = dir2 + '/crop_' + self.frame
        cv2.imwrite(filename, crop)

    #----------------------------------------------------------------------
    def nextPicture(self):
        """
        Loads the next picture in the directory
        """
        if (self.rowNumber < (len(self.csvRows)-1)):
            self.rowNumber += 1

        self.line = self.csvRows[self.rowNumber]
        self.setImageParameters(self.line)
        self.currentImage = self.imagePath + '/' + self.frame

        self.loadImage(self.currentImage)

        if (self.cropOnNext):
            self.cropPicture()

    def previousPicture(self):
        """
        Loads the next picture in the directory
        """
        if (self.rowNumber > 0):
            self.rowNumber -= 1

        self.line = self.csvRows[self.rowNumber]
        self.setImageParameters(self.line)
        self.currentImage = self.imagePath + '/' + self.frame

        self.loadImage(self.currentImage)

    #----------------------------------------------------------------------
    def update(self, event):
        """
        Called when the slideTimer's timer event fires. Loads the next
        picture from the folder by calling th nextPicture method
        """
        self.nextPicture()

    #----------------------------------------------------------------------
    def loadTrainingCSV(self, msg):
        self.imagePath = msg;
        self.trainFilename = self.imagePath+'/train.csv'

        with open(self.trainFilename, 'r') as csvfile:
            reader = csv.DictReader(csvfile, fieldnames=self.csvFields)
            self.csvRows = list(reader)

        """
        self.csvreader = csv.DictReader(self.csvFile, fieldnames=self.csvFields)
        self.csvwriter = csv.DictWriter(self.tempCsvFile, fieldnames=self.csvFields)

        for x in range(0,self.startingRow):
            self.line = next(self.csvreader)
            self.csvwriter.writerow(self.line)
            self.rowNumber += 1
        """
        self.rowNumber = self.startingRow
        while True:
            self.line = self.csvRows[self.rowNumber]
            if int(self.line[self.csvFields[3]]) < 91:
                break
            self.rowNumber += 1

        self.setImageParameters(self.line)
        self.currentImage = self.imagePath + '/' + self.frame
        self.loadImage(self.currentImage)

    #----------------------------------------------------------------------
    def setImageParameters(self, line):
        self.frame = line[self.csvFields[0]]
        self.xmin = int(line[self.csvFields[4]])
        self.xmax = int(line[self.csvFields[6]])
        self.ymin = int(line[self.csvFields[5]])
        self.ymax = int(line[self.csvFields[7]])
        self.classification = int(line[self.csvFields[3]])

    #----------------------------------------------------------------------
    def onNext(self, event):
        """
        Calls the nextPicture method
        """
        if self.classification < 91:
            # default to buddy class
            self.newClassification = self.class_labels_rev['buddy']
            self.classification = self.newClassification
            print(self.classification)

        self.line[self.csvFields[3]] = self.classification
        self.nextPicture()

    #----------------------------------------------------------------------
    def onCrop(self, event):
        """
        Calls the nextPicture method
        """
        self.cropPicture()

    #----------------------------------------------------------------------
    def onPrevious(self, event):
        """
        Calls the previousPicture method
        """
        self.previousPicture()

    #----------------------------------------------------------------------
    def onUpdateCSV(self, event):
        """
        Opens a DirDialog to allow the user to open a folder with pictures
        """

        """ default to Jade instead of popping up dialog
        dlg = MyDialog()

        if dlg.ShowModal() == wx.ID_OK:
            self.newClassification = self.class_labels_rev[dlg.comboBox1.GetValue()]
            self.line[self.csvFields[3]] = self.newClassification
            print(self.newClassification)
            self.setImageParameters(self.line)
            self.loadImage(self.currentImage)
        """

        self.newClassification = self.class_labels_rev['jade']
        self.line[self.csvFields[3]] = self.newClassification
        print(self.newClassification)
        self.setImageParameters(self.line)
        self.loadImage(self.currentImage)

    def onDone(self, event):
        tempCsvFile = NamedTemporaryFile(mode='w', delete=False, newline='')
        csvwriter = csv.DictWriter(tempCsvFile, fieldnames=self.csvFields)

        for row in self.csvRows:
            csvwriter.writerow(row)
        tempCsvFile.close()

        shutil.move(tempCsvFile.name, self.trainFilename)

        #quit()

    def onValidate(self, event):
        self.train()

    def train(self):
        img_lst = []
        for row in self.csvRows:
            if row[self.csvFields[5]] != 'class_id' and int(row[self.csvFields[5]]) > 0:
                img_lst.append((row[self.csvFields[0]],row[self.csvFields[1]],row[self.csvFields[2]],row[self.csvFields[3]],row[self.csvFields[4]],row[self.csvFields[5]]))

        random.shuffle(img_lst)
        img_lst=np.array(img_lst)

        """
        if self.stratified:
            #from sklearn.model_selection import StratifiedShuffleSplit
            from sklearn.model_selection import train_test_split

            ## Stratified sampling to generate train and validation sets
            labels_train=img_lst[:,5]
            # unique_train, counts_train = np.unique(labels_train, return_counts=True) # To have a look at the frecuency distribution
            #sss = StratifiedShuffleSplit(train_size=1-self.percentVal, test_size=self.percentVal, random_state=0)
            sss = train_test_split(labels_train, 1, test_size=self.percentVal)

            for tr_idx, va_idx in sss:
                print("Train subset has ", len(tr_idx), " cases. Validation subset has ", len(va_idx), "cases")
        else:
        """
        (nRows, nCols) = img_lst.shape
        splitat=int(round(nRows*(1-self.percentVal),0))
        tr_idx=range(0,splitat)
        va_idx=range(splitat,nRows)
        print("Train subset has ", len(tr_idx), " cases. Validation subset has ", len(va_idx), "cases")

        tr_lst=img_lst[tr_idx,:].tolist()
        va_lst=img_lst[va_idx,:].tolist()

        trainFile = open(self.imagePath+'/training.csv', 'w', newline='')
        csvTrainWriter = csv.writer(trainFile)
        valildationFile = open(self.imagePath+'/validation.csv', 'w', newline='')
        csvValidationWriter = csv.writer(valildationFile)

        csvTrainWriter.writerow(self.csvFields)
        csvValidationWriter.writerow(self.csvFields)
        for item in tr_lst:
            csvTrainWriter.writerow(item)
        for item in va_lst:
            csvValidationWriter.writerow(item)


########################################################################
class MyDialog(wx.Dialog):
    """"""

    #----------------------------------------------------------------------
    def __init__(self):
        """Constructor"""
        wx.Dialog.__init__(self, None, title="Update classification")

        self.comboBox1 = wx.ComboBox(self,
                                     choices=['background', 'buddy', 'jade'],
                                     value="")
        okBtn = wx.Button(self, wx.ID_OK)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.comboBox1, 0, wx.ALL|wx.CENTER, 5)
        sizer.Add(okBtn, 0, wx.ALL|wx.CENTER, 5)
        self.SetSizer(sizer)

########################################################################
class ViewerFrame(wx.Frame):
    """"""

    #----------------------------------------------------------------------
    def __init__(self):
        """Constructor"""
        wx.Frame.__init__(self, None, title="Image Viewer")
        panel = ViewerPanel(self)
        self.folderPath = ""
        pub.subscribe(self.resizeFrame, ("resize"))

        self.initToolbar()
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(panel, 1, wx.EXPAND)
        self.SetSizer(self.sizer)

        self.Show()
        self.sizer.Fit(self)
        self.Center()


    #----------------------------------------------------------------------
    def initToolbar(self):
        """
        Initialize the toolbar
        """
        self.toolbar = self.CreateToolBar()
        self.toolbar.SetToolBitmapSize((16,16))

        open_ico = wx.ArtProvider.GetBitmap(wx.ART_FILE_OPEN, wx.ART_TOOLBAR, (16,16))
        openTool = self.toolbar.AddSimpleTool(wx.ID_ANY, open_ico, "Open", "Open an Image Directory")
        self.Bind(wx.EVT_MENU, self.onOpenDirectory, openTool)

        self.toolbar.Realize()

    #----------------------------------------------------------------------
    def onOpenDirectory(self, event):
        """
        Opens a DirDialog to allow the user to open a folder with pictures
        """
        dlg = wx.DirDialog(self, "Choose a training csv",
                           style=wx.DD_DEFAULT_STYLE)

        if dlg.ShowModal() == wx.ID_OK:
            self.folderPath = dlg.GetPath()
            print (self.folderPath)
        pub.sendMessage(topicName="update images", msg=self.folderPath)
        
    #----------------------------------------------------------------------
    def resizeFrame(self, msg):
        """"""
        self.sizer.Fit(self)
        
#----------------------------------------------------------------------
if __name__ == "__main__":
    app = wx.App()
    frame = ViewerFrame()
    app.MainLoop()
    