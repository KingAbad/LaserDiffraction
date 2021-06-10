import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from tqdm import tqdm
import tkinter as tk
from tkinter import *
from tkinter.filedialog import askopenfilename
import numpy as np
import time
import os
from PIL import Image, ImageTk, ImageOps
import cv2
try:
    import miepython

except ModuleNotFoundError:
    print('miepython not installed. To install, uncomment and run the cell above.')
    print('Once installation is successful, rerun this cell again.')

global oldclick
oldclick = (0, 0)
global norr
global useold

#norr = 1
#useold = False

def getImage():#func to get img
    coord = []
    global file
    file = askopenfilename(initialdir="image/",title='Choose an image.')
    global filename
    filename = file.split('/')[-1]

def gotodataapp(imported = False, imgimport = None):#startup function containing all widgets for tkinter and their other funcs
    root = tk.Tk()
    global oldclick
    global imgimport1
    imgimport1 = imgimport
    print("data")
    #root = tk.Tk()

    root.title("Abad Sethi's MiePlot app")
    root.geometry("600x200")


    global useold
    global norr
    norr = IntVar()
    norr.set(0)
    useold = BooleanVar()
    useold.set(False)

    def swapbool():#these two funcs fix a garbage collection bug created by the function structure of this tkinter section by taking the checkbutton var handling
        if (useold.get()):
            useold.set(False)
        else:
            useold.set(True)
        print(useold.get())
    def swapnum():
        if norr.get() == 1:
            norr.set(0)
        else:
            norr.set(1)
        print(norr.get())
        
    
    global refve
    global refrve
    global sige
    global mue
    global distse
    global thete
    global pixelsze
    global imgDiste
    global lame
    global enam
    global sdege
    global edege
    refv = tk.Label(root, text="reference value :").grid(row=1,column=1)
    refve = tk.Entry(root)
    refve.insert(END,'20000e-9')
    refve.grid(row=1,column=2)
    refrv = tk.Label(root, text="refractive index :").grid(row=1,column=3)
    refrve = tk.Entry(root)
    refrve.insert(END,'1.33')
    refrve.grid(row=1,column=4)
    lam = tk.Label(root, text="value of lambda :").grid(row=2,column=1)
    lame = tk.Entry(root)
    lame.insert(END,'632e-9')
    lame.grid(row=2,column=2)
    sig = tk.Label(root, text="sigma :").grid(row=2,column=3)
    sige = tk.Entry(root)
    sige.insert(END,'20000e-9')
    sige.grid(row=2,column=4)
    mu = tk.Label(root, text="mu :").grid(row=3,column=1)
    mue = tk.Entry(root)
    mue.insert(END,'20000e-9')
    mue.grid(row=3,column=2)
    nord = tk.Label(root, text="normal distributon? :").grid(row=3,column=3)
    norde = tk.Checkbutton(root, variable=norr, onvalue=1, offvalue=0, command=swapnum)
    norde.grid(row=3,column=4)
    norde.var = norr
    dists = tk.Label(root, text="distribution size :").grid(row=4,column=1)
    distse = tk.Entry(root)
    distse.insert(END,'10000')
    distse.grid(row=4,column=2)
    thet = tk.Label(root, text="theta size :").grid(row=4,column=3)
    thete = tk.Entry(root)
    thete.insert(END,'100')
    thete.grid(row=4,column=4)
    sdeg = tk.Label(root, text="start value (degrees) :").grid(row=5,column=1)
    sdege = tk.Entry(root)
    sdege.insert(END,'0')
    sdege.grid(row=5,column=2)
    edeg = tk.Label(root, text="end value (degrees) :").grid(row=5,column=3)
    edege = tk.Entry(root)
    edege.insert(END,'20')
    edege.grid(row=5,column=4)
    pixelsz = tk.Label(root, text="Pixelsize :").grid(row=6,column=1)
    pixelsze = tk.Entry(root)
    pixelsze.insert(END,'0.00005')
    pixelsze.grid(row=6,column=2)
    imgDist = tk.Label(root, text="Distance in m :").grid(row=6,column=3)
    imgDiste = tk.Entry(root)
    imgDiste.insert(END,'1.64')
    imgDiste.grid(row=6,column=4)
    addImgBtn = Button(root, text="Choose an image", command=getImage).grid(row=8,column=3)
    nam = tk.Label(root, text="parameter save name :").grid(row=7,column=1)
    enam = tk.Entry(root)
    enam.grid(row=7,column=2)
    use = tk.Label(root, text="use previous centre :").grid(row=7,column=3)
    usee = tk.Checkbutton(root, variable=useold, onvalue=True, offvalue=False, command=swapbool)
    usee.grid(row=7,column=4)
    usee.var = useold
    def Clear():
        refve.delete(0, "end")
        refrve.delete(0, "end")
        lame.delete(0, "end")
        sige.delete(0, "end")
        mue.delete(0, "end")
        distse.delete(0, "end")
        thete.delete(0, "end")
        enam.delete(0, "end")
        edege.delete(0, "end")
        sdege.delete(0, "end")
        pixelsze.delete(0, "end")
        imgDiste.delete(0, "end")


    def staggerbegin():
        #print(useold.get())
        #print(norr.get())
        Begin(imported, imgimport)
        
    
    clearbutton = tk.Button(root, text="clear", command=Clear).grid(row=8,column=2)
    gobut = tk.Button(root, text="Start", command=staggerbegin).grid(row=8,column=1)
    root.mainloop()
    
def Begin(imported = False, imgimport = None):#begin contains program logic
    global oldclick
    global useold
    global norr
    #global useold
    fontSize = 10;
    plt.rcParams.update({'font.size': fontSize})
    if imported == False:
        raw_path = os.path.normpath(file)
        Image = plt.imread(raw_path)#imfe.get())#filenames[0])#'D:\\Users\\Welcome\\Downloads\\RingPictures.jpg')
    else:
        Image = imgimport1
    
    h, w, _ = Image.shape
    imagepixelwidth = w
    global pixelsze
    global imgDiste
    pixelsize = float(pixelsze.get())#int(input('Please enter the pixelsize: '))
    imagedistance = float(imgDiste.get())#int(input('Please enter the distance in m: '))
    anglePixel = np.arctan(pixelsize/imagedistance)*180/np.pi

    grayImage = rgb2gray(Image)*255;

    fig = plt.figure(figsize=(20,20));

    axc = fig.add_subplot(2,2,1)

    axc.imshow(grayImage,cmap='gray');
    axc.set_title('Original Grayscale Image');

    # fig2=plt.figure(2)
    axc.text(-0.1,-0.1,"click in the center of the diffraction pattern")
    #plt.pause(2)
    # plt.close(fig2)
    
    if useold.get():
        x = oldclick[0]
        y = oldclick[1]
    else:
        xy= fig.ginput(1, show_clicks=True);
        x=xy[0][0]
        y=xy[0][1]

    
    oldclick = (x, y)
    print("useold = " + str(useold.get()))
    print("oldclick = " + str(oldclick))
    #print(xy)

    xCenter = x;
    yCenter = y;

    conemultiplier = 5232 / imagepixelwidth
    
    N = 1600 / conemultiplier#1600 for 5232 by 3488 pixels
    N = int(N)
    intensity=np.zeros(N)
    intensity2=np.zeros(N)


    for radius in tqdm(range(1,N +1)):
    # 	print(radius)

            theta = np.deg2rad(np.linspace(-20, 20, 10)%360);
            theta2 = np.deg2rad(np.linspace(150, 210 , 10));
            x1= radius * np.cos(theta) + xCenter; 
            y1 = radius * np.sin(theta) + yCenter; 
            x2= radius * np.cos(theta2) + xCenter; 
            y2 = radius * np.sin(theta2)+ yCenter; 

            axc.plot(x1, y1, 'b-', linewidth=1);
            axc.plot(x2, y2, 'r-', linewidth=1); 

            profile=np.zeros(len(x1)) 

            for k in range(len(x1)):
                    profile[k] = grayImage[int(round(y1[k])), int(round(x1[k]))]; #logical indexing

            intensity[radius-1]=np.mean(profile);  

            profile2=np.zeros(len(x2)) 

            for z in range(len(x2)):
                    profile2[z] = grayImage[int(round(y2[z])), int(round(x2[z]))]; # logical indexing

            intensity2[radius-1]=np.mean(profile2);

    angleXaxis1 = np.linspace(0,len(intensity)*anglePixel,len(intensity));


    axc = fig.add_subplot(6,2,2)

    axc.plot(angleXaxis1,intensity, 'b-')
    axc.grid(True)
    axc.set_title('Intensity profile blue');
    axc.set_xlabel('Angle [degrees]');
    axc.set_ylabel('Intensity');

    axc = fig.add_subplot(6,2,4)

    angleXaxis2 = np.linspace(0,len(intensity2)*anglePixel,len(intensity2));
    axc.plot(angleXaxis2,intensity2, 'r-')
    axc.grid(True);
    axc.set_title('Intensity profile red');
    axc.set_xlabel('Angle [degrees]');
    axc.set_ylabel('Intensity');

    #%%###########################################################################
    #%%################################# distr code ##############################
    #%%###########################################################################

    # fncs that may or may not be useful

    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'same') / w

    def lognstat(mu, sigma):
        """Calculate the mean of and variance of the lognormal distribution given
        the mean (`mu`) and standard deviation (`sigma`), of the associated normal 
        distribution."""
        m = np.exp(mu + sigma**2 / 2.0)
        v = np.exp(2 * mu + sigma**2) * (np.exp(sigma**2) - 1)
        return m, v
    #%% Read/load parameter values + distribution
    ''' reference d value '''
    #d_ref = 20000e-9

    ''' let the user decide'''
    global refve
    global refrve
    global sige
    global mue
    global distse
    global lame
    global thete
    d_ref = float(refve.get()) #float(input('Please input the reference value of d: '))
    ref_Index = float(refrve.get()) #float(input('Please input the refractive index: '))
    wafeForm = float(lame.get())#float(input('Please input the value of lambda: '))
    sigma_distr = float(sige.get())#float(input('Please enter sigma: '))
    mu_distr = float(mue.get())#float(input('Please input mu: '))
    distr = norr.get()#int(input('Please choose the distribution:\n1:\tNormal\t2:\tLognormal\n'))
    size_distr = int(distse.get())#int(input('Please enter the size of distribution: '))
    theta_var = int(thete.get())#int(input('Please enter the size of theta: '))
    #while distr!=1 and distr!=2:
    #    distr = int(input('Invalid choice!\nPlease choose the distribution:\n1:\tNormal\t2:\tLognormal\n'))
    global enam
    with open("args.txt", "a") as f:
        f.write("""
    """)
        f.write("with the save name of :" + str(enam.get()) + " at " + str(time.localtime(time.time())) + "these parameters were used :"
                +"reference value of : " + str(refve.get())
                +"refractive index of : " + str(refrve.get())
                +"lambda value of : " + str(lame.get())
                +"sigma value of : " + str(sige.get())
                +"mu value of : " + str(mue.get())
                +"normal distribution of (1 = normal) : " + str(norr.get())
                +"distribution size value of : " + str(distse.get())
                +"theta value of : " + str(thete.get())

    )

    sigma_distr_Var = 10/100*sigma_distr

    ''' try normal'''
    #mu_distr, sigma_distr = (d_ref, 10/100*d_ref)
    #distr = 1
    #size_distr = int(10)


    '''try lognormal'''
    #mu_distr, sigma_distr = (d_ref, 10/100*d_ref)
    #distr = 2
    #size_distr = int(1e3)

    ''' param definitions '''
    global sdege
    global edege
    m = ref_Index-0j
    lambda0 = wafeForm  # m = 632e-9
    theta = np.linspace(int(sdege.get()),int(edege.get()),theta_var)############################################################################################ previously 0 and 10 and theta_var
    mu = np.cos(theta* np.pi/180)

    ''' choose the distribution '''
    if distr == 1:
        d_orig = np.random.normal(mu_distr, sigma_distr_Var, size=size_distr) # m; Normal
        bins_h = np.histogram_bin_edges(np.random.normal(mu_distr, sigma_distr_Var/np.sqrt(2*np.pi), size=int(1e6)), size_distr)
        d = bins_h[:-1]
    elif distr == 0:
        d_orig = np.random.lognormal(mu_distr, sigma_distr_Var, size=size_distr)  # m; Logormal
        bins_h = np.histogram_bin_edges(np.random.lognormal(mu_distr, sigma_distr_Var/np.sqrt(2*np.pi), size=int(1e6)), size_distr)
        d = bins_h[:-1]
    #%% Calcualtion part
    ''' ref '''
    x = 2 * np.pi/lambda0 * d_ref/2
    geometric_cross_section = np.pi * d_ref**2/4 * 1e4  # cm**2
    qext, qsca, qback, g = miepython.mie(m,x)
    sigma_sca_ref = geometric_cross_section * qext * miepython.i_unpolarized(m,x,mu)

    ''' distr '''
    x = 2 * np.pi/lambda0 * d/2
    geometric_cross_section = np.pi * d**2/4 * 1e4  # cm**2
    sigma_sca = np.zeros(len(theta))
    for idx in tqdm(range(size_distr)):
        qext, qsca, qback, g = miepython.mie(m,x[idx])
        sigma_sca += 1/size_distr*(geometric_cross_section[idx] * qext * miepython.i_unpolarized(m,x[idx],mu))
    #%% plotting part
    fig, ax = plt.subplots(1,2,figsize=(15,5))

    ax[0].semilogy(theta, sigma_sca_ref*1e-3,
                   color='tab:blue', label="%.0fnm\n(x10$^{-3}$)" % (d_ref*1e9))
    ax[0].semilogy(theta, moving_average(sigma_sca, 1)*1e-3, color='tab:orange',
                     label="mean:%.0fnm\n(x10$^{-3}$)" % (np.mean(d)*1e9))
    ax[0].legend()
    ax[0].set_title("Refractive index m=" + str(refrve.get()) + ", $\lambda$=" + str(lame.get()) + "nm & Intensity profile blue")
    ax[0].set_xlabel("Scattering Angle (degrees)")
    ax[0].set_ylabel("Diff. Scattering Cross Section (cm$^2$/sr)")
    ax[0].grid(True)

    ax2 = ax[0].twinx()  # instantiate a second axes that shares the same x-axis
    angleXaxis2_temp = np.linspace(theta[0],theta[-1],len(angleXaxis2));
    ax2.plot(angleXaxis2_temp, intensity2*10000    , 'r-', label='Intensity profile red')
    ax2.set_ylabel('Intensity');
    ax2.legend()

    count_h, bins_h, _ = ax[1].hist(d_orig, 30, density=True, label="Histogram,\n$\mu$=%.2g $\sigma$=%.2g" % (np.mean(d_orig), np.std(d_orig)))
    if distr == 1:
        # Normal
        pdf_distr = 1/(sigma_distr_Var * np.sqrt(2 * np.pi)) *\
             np.exp( - (bins_h - mu_distr)**2 / (2 * sigma_distr_Var**2))
        title_str = " (Normal Distribution)"
    elif distr == 0:
        # Logormal
        pdf_distr = (np.exp(-(np.log(bins_h) - mu_distr)**2 / (2 * sigma_distr_Var**2))/\
                 (bins_h * sigma_distr_Var * np.sqrt(2 * np.pi)))
        title_str = " (Lognormal Distribution)"
    ax[1].plot(bins_h, pdf_distr, linewidth=2, color='tab:red',
               label="Theoretical pdf,\n$\mu$=%.2g $\sigma$=%.2g" % (mu_distr, sigma_distr_Var))
    ax[1].set_title("Histogram vs PDF of d" + title_str)
    ax[1].set_xlabel("d (m)")
    ax[1].set_ylabel("Density")
    ax[1].grid(True)
    ax[1].legend()

    plt.show()


        


















global darkphotocounter

def gotosizeapp():
    print("size")
    nroot = tk.Tk()
    nroot.geometry("455x100")
    nroot.resizable(False, False)
    label1 = tk.Label(nroot, text="""amount of dark photos that align with this light photo
(you can select one light photo then this amount of dark photos straight afterwards):""").place(relx=0.5, rely=0.25, anchor=CENTER)
    global darkphotocounter
    darkphotocounter = tk.Entry(nroot)
    darkphotocounter.place(relx=0.5, rely=0.5, anchor=CENTER)
    startbtn2 = tk.Button(nroot, text="select images", command=beginsizeapp).place(relx=0.5, rely=0.75, anchor=CENTER)

    
    nroot.mainloop()

def beginsizeapp():
    class PerspectiveTransform():
        def __init__(self, master):
            global darkphotocounter
            self.dpc = int(darkphotocounter.get())#dpc stands for dark photo count
            self.iter = 0
            self.startarea = True
            self.parent = master
            self.coord = [] 	# x,y coordinate
            self.dot = []
            self.reffile = '' 	 	#reference image path
            self.reffilename ='' 	#reference image filename
            self.files = []                 #dark image paths
            self.filenames = []           #dark image filenames
            
            
            #setting up a tkinter canvas with scrollbars
            self.frame = Frame(self.parent, bd=2, relief=SUNKEN)
            self.frame.grid_rowconfigure(0, weight=1)
            self.frame.grid_columnconfigure(0, weight=1)
            self.xscroll = Scrollbar(self.frame, orient=HORIZONTAL)
            self.xscroll.grid(row=1, column=0, sticky=E+W)
            self.yscroll = Scrollbar(self.frame)
            self.yscroll.grid(row=0, column=1, sticky=N+S)
            self.canvas = Canvas(self.frame, bd=0, xscrollcommand=self.xscroll.set, yscrollcommand=self.yscroll.set)
            self.canvas.grid(row=0, column=0, sticky=N+S+E+W)
            self.xscroll.config(command=self.canvas.xview)
            self.yscroll.config(command=self.canvas.yview)
            self.frame.pack(fill=BOTH,expand=1)
            self.addrefImage()
            
            #mouseclick event and button
            self.canvas.bind("<Button 1>",self.insertCoords)
            self.canvas.bind("<Button 3>",self.removeCoords)
            self.ctrPanel = Frame(self.frame)
            self.ctrPanel.grid(row = 0, column = 2, columnspan = 2, sticky = N+E)
            self.addImgBtn = Button(self.ctrPanel, text="Upload een foto", command=self.addrefImage)
            self.addImgBtn.grid(row=0,column=2, pady = 5, sticky =NE)
            self.saveBtn = Button(self.ctrPanel, text="Save", command=self.saveImage)
            self.saveBtn.grid(row=1,column=2, pady = 5, sticky =NE)
            self.portBtn = Button(self.ctrPanel, text="use this image in data app", command=self.porting)######remember to check back on port
            self.portBtn.grid(row=2,column=2, pady = 5, sticky =NE)
            self.nextBtn = Button(self.ctrPanel, text="next dark image", command=self.nextimg)
            self.nextBtn.grid(row=3,column=2, pady = 5, sticky =NE)
            self.canvas.bind('<Configure>', self._resize_image)



        def _resize_image(self,event):
            if self.startarea:
                new_width = event.width
                new_height = event.height

                self.sizedimgtemp = ImageOps.pad(self.imgtemp, size=(new_width, new_height))
                self.img = ImageTk.PhotoImage(self.sizedimgtemp, master = self.canvas)

                self.canvas.create_image(0,0,image=self.img,anchor="nw")
            
        #adding the image
        def addrefImage(self):
            self.startarea = True
            self.coord = []
            self.reffile = askopenfilename(parent=self.parent, initialdir="image/",title='Choose a reference image.')
            self.reffilename = self.reffile.split('/')[-1]
            self.reffilename = self.reffilename.rstrip('.jpg')
            self.imgtemp = Image.open(self.reffile)
            self.img = ImageTk.PhotoImage(self.imgtemp, master = self.canvas)
            self.canvas.create_image(0,0,image=self.img,anchor="nw")
            self.canvas.config(scrollregion=self.canvas.bbox(ALL), width=self.img.width(), height=self.img.height())
            if self.dpc > 1: 
                for i in range(1, self.dpc + 1):
                    self.addImage()
            else:
                self.addImage()

        def addImage(self):
            self.coord = []
            self.files.append(askopenfilename(parent=self.parent, initialdir="image/",title='Choose a dark image.'))
            templ = self.files[-1].split('/')[-1]
            self.filenames.append(templ.rstrip('.jpg'))
        
        #Save coord according to mouse left click
        def insertCoords(self, event):
            #outputting x and y coords to console
            self.coord.append([event.x, event.y])
            r=3
            self.dot.append(self.canvas.create_oval(event.x - r, event.y - r, event.x + r, event.y + r, fill="#ff0000"))         #print circle
            if (len(self.coord) == 4):
                #if self.dpc > 1: 
                #    for i in range(1, self.dpc + 1):
                #        self.Transformer(self.files[i-1], self.filenames[i-1])
                #else:
                self.Transformer(self.files[0], self.filenames[0])
                self.iter += 1
                self.startarea = False
                #self.Transformer()
                self.canvas.delete("all")
                self.canvas.create_image(0,0,image=self.result,anchor="nw")
                self.canvas.image = self.result
        
        #remove last inserted coord using mouse right click
        def removeCoords(self, event=None):
            del self.coord[-1]
            self.canvas.delete(self.dot[-1])
            del self.dot[-1]
        
        def Transformer(self, currentdark, currentname):
            width_factor = self.imgtemp.size[0] / self.sizedimgtemp.size[0]
            height_factor = self.imgtemp.size[1] / self.sizedimgtemp.size[1]#get resized change as a coefficient
            for i in range(4):
                self.coord[i] = [int(self.coord[i][0] * width_factor), int(self.coord[i][1] * height_factor)]
            print(self.coord)
            frame = cv2.imread(currentdark)#self.file)
            frame_circle = frame.copy()
            #points = [[480,90],[680,90],[0,435],[960,435]]
            cv2.circle(frame_circle, tuple(self.coord[0]), 5, (0, 0, 255), -1)
            cv2.circle(frame_circle, tuple(self.coord[1]), 5, (0, 0, 255), -1)
            cv2.circle(frame_circle, tuple(self.coord[2]), 5, (0, 0, 255), -1)
            cv2.circle(frame_circle, tuple(self.coord[3]), 5, (0, 0, 255), -1)
            
            widthA = np.sqrt(((self.coord[3][0] - self.coord[2][0]) ** 2) + ((self.coord[3][1] - self.coord[2][1]) ** 2))
            widthB = np.sqrt(((self.coord[1][0] - self.coord[0][0]) ** 2) + ((self.coord[1][1] - self.coord[0][1]) ** 2))
            maxWidth = max(int(widthA), int(widthB))
             
            heightA = np.sqrt(((self.coord[1][0] - self.coord[3][0]) ** 2) + ((self.coord[1][1] - self.coord[3][1]) ** 2))
            heightB = np.sqrt(((self.coord[0][0] - self.coord[2][0]) ** 2) + ((self.coord[0][1] - self.coord[2][1]) ** 2))
            maxHeight = max(int(heightA), int(heightB))
         
            print(self.coord)
            pts1 = np.float32(self.coord)    
            pts2 = np.float32([[0, 0], [maxWidth-1, 0], [0, maxHeight-1], [maxWidth-1, maxHeight-1]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            self.result_cv = cv2.warpPerspective(frame, matrix, (maxWidth,maxHeight))
             
            #cv2.imshow("Frame", frame_circle)
            #cv2.imshow("Perspective transformation", result_cv)
            
            result_rgb = cv2.cvtColor(self.result_cv, cv2.COLOR_BGR2RGB)
            self.result = ImageTk.PhotoImage(image = Image.fromarray(result_rgb), master = self.canvas)
            
        def saveImage(self):
            cv2.imwrite("result/"+currentname+"_res.jpg", self.result_cv)#used to ve self.filename
            print(self.filename+" is saved!")
        def porting(self):
            gotodataapp(True, imgimport=self.result_cv)
        def nextimg(self):
            if self.iter < len(self.files):
                #self.canvas.delete("all")
                self.Transformer(self.files[self.iter], self.filenames[self.iter])
                self.iter += 1
                self.canvas.delete("all")
                self.canvas.create_image(0,0,image=self.result,anchor="nw")
                self.canvas.image = self.result
    #---------------------------------
    if __name__ == '__main__':
        qroot = Tk()
        #root.geometry("1360x740")
        transformer = PerspectiveTransform(qroot)
        qroot.mainloop()






mroot = tk.Tk()
mroot.resizable(False, False) 
mroot.title("Abad Sethi's Mieplot App")
mroot.geometry("300x100")

sizeapp = tk.Button(mroot, text="size app", command=gotosizeapp).place(x=120, rely=0.5, anchor=CENTER)
dataapp = tk.Button(mroot, text="data app", command=gotodataapp).place(x=180, rely=0.5, anchor=CENTER)



mroot.mainloop()
