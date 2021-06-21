import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from tqdm import tqdm
import tkinter as tk
from tkinter import *
from tkinter.filedialog import askopenfilename, asksaveasfile
import numpy as np
import time
import os
from decimal import *
import math
from PIL import Image, ImageTk, ImageOps
import cv2
from mpl_axes_aligner import shift
#from scipy.interpolate import interp1d
from scipy.interpolate import make_interp_spline

try:
    import miepython

except ModuleNotFoundError:
    print('miepython not installed. To install, uncomment and run the cell above.')
    print('Once installation is successful, rerun this cell again.')
try:
    os.mkdir(os.getcwd() + "\\result")
except:
    print()
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


def getrefImage():#func to get img
    coord = []
    global reffile
    reffile = askopenfilename(initialdir="image/",title='Choose an image.')
    global reffilename
    reffilename = reffile.split('/')[-1]

def gotodataapp(imported = False, imgimport = None, refimgimport = None):#startup function containing all widgets for tkinter and their other funcs
    root = tk.Tk()
    root.resizable(False, False) 
    global oldclick
    global imgimport1
    imgimport1 = imgimport
    print("data")
    #root = tk.Tk()

    root.title("Abad Sethi's MiePlot app")
    root.geometry("580x220")


    global useold
    global norr
    norr = IntVar()
    norr.set(1)
    useold = BooleanVar()
    useold.set(False)
    userough = BooleanVar()
    userough.set(True)
    usecurved = BooleanVar()
    usecurved.set(True)

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

    def swapboolrough():
        if (userough.get()):
            userough.set(False)
        else:
            userough.set(True)
        print(userough.get())

    def swapboolcurved():
        if (usecurved.get()):
            usecurved.set(False)
        else:
            usecurved.set(True)
        print(usecurved.get())
        
    
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
    global ymulte
    global xmulte
    global roughe
    global curvede
    refv = tk.Label(root, text="Reference value (nm) :").grid(row=1,column=1)
    refve = tk.Entry(root)
    refve.grid(row=1,column=2)
    refve.insert(END,'100000e-9')
    
    refrv = tk.Label(root, text="Refractive index :").grid(row=1,column=3)
    refrve = tk.Entry(root)
    refrve.grid(row=1,column=4)
    refrve.insert(END,'1.33')
    
    lam = tk.Label(root, text="Value of lambda (nm) :").grid(row=2,column=1)
    lame = tk.Entry(root)
    lame.insert(END,'632e-9')
    lame.grid(row=2,column=2)
    
    sig = tk.Label(root, text="Sigma (nm) :").grid(row=2,column=3)
    sige = tk.Entry(root)
    sige.grid(row=2,column=4)
    sige.insert(END,'100000e-9')
    
    mu = tk.Label(root, text="Mu (nm) :").grid(row=3,column=1)
    mue = tk.Entry(root)
    mue.grid(row=3,column=2)
    mue.insert(END,'100000e-9')
    
    nord = tk.Label(root, text="Normal distributon? :").grid(row=3,column=3)
    norde = tk.Checkbutton(root, variable=norr, onvalue=1, offvalue=0, command=swapnum)
    norde.grid(row=3,column=4)
    norde.select()
    norde.var = norr
    
    dists = tk.Label(root, text="Amount of dist clc points :").grid(row=4,column=1)
    distse = tk.Entry(root)
    distse.grid(row=4,column=2)
    distse.insert(END,'1000')
    
    thet = tk.Label(root, text="Angle clc points :").grid(row=4,column=3)
    thete = tk.Entry(root)
    thete.grid(row=4,column=4)
    thete.insert(END,'100')
    
    #sdeg = tk.Label(root, text="start value (degrees) :").grid(row=5,column=1)
    #sdege = tk.Entry(root)
    #sdege.grid(row=5,column=2)
    #edeg = tk.Label(root, text="end value (degrees) :").grid(row=5,column=3)
    #edege = tk.Entry(root)
    #edege.grid(row=5,column=4)
    
    pixelsz = tk.Label(root, text="Pixelsize :").grid(row=5,column=1)
    pixelsze = tk.Entry(root)
    pixelsze.grid(row=5,column=2)
    pixelsze.insert(END,'0.00007')
    
    imgDist = tk.Label(root, text="Distance (m) :").grid(row=5,column=3)
    imgDiste = tk.Entry(root)
    imgDiste.grid(row=5,column=4)
    imgDiste.insert(END,'1.45')
    
    addImgBtn = Button(root, text="Upload an image", command=getImage).grid(row=9,column=3)
    addrefImgBtn = Button(root, text="Upload a reference image", command=getrefImage).grid(row=9,column=4)
    
    nam = tk.Label(root, text="parameter save name :").grid(row=8,column=1)
    enam = tk.Entry(root)
    enam.grid(row=8,column=2)
    
    use = tk.Label(root, text="Use previous centre :").grid(row=8,column=3)
    usee = tk.Checkbutton(root, variable=useold, onvalue=True, offvalue=False, command=swapbool)
    usee.grid(row=8,column=4)
    #usee.select()
    usee.var = useold

    rough = tk.Label(root, text="Display rough graph :").grid(row=7,column=1)
    roughe = tk.Checkbutton(root, variable=userough, onvalue=True, offvalue=False, command=swapboolrough)
    roughe.grid(row=7,column=2)
    roughe.select()
    roughe.var = userough

    curved = tk.Label(root, text="Display smoothened graph :").grid(row=7,column=3)
    curvede = tk.Checkbutton(root, variable=usecurved, onvalue=True, offvalue=False, command=swapboolcurved)
    curvede.grid(row=7,column=4)
    curvede.select()
    curvede.var = usecurved
    
    xmult = tk.Label(root, text="Smoothness :").grid(row=6,column=1)
    xmulte = tk.Entry(root)
    xmulte.grid(row=6,column=2)
    xmulte.insert(END,'20')
    
    ymult = tk.Label(root, text="Y-offset :").grid(row=6,column=3)
    ymulte = tk.Entry(root)
    ymulte.grid(row=6,column=4)
    ymulte.insert(END,'150')
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
        #sdege.delete(0, "end")
        pixelsze.delete(0, "end")
        imgDiste.delete(0, "end")
        #ymulte.delete(0, "end")
        xmulte.delete(0, "end")
        ymulte.delete(0, "end")


    def staggerbegin():
        #print(useold.get())
        #print(norr.get())
        Begin(imported, imgimport, refimgimport)
        
    
    clearbutton = tk.Button(root, text="clear", command=Clear).grid(row=9,column=2)
    gobut = tk.Button(root, text="Start", command=staggerbegin).grid(row=9,column=1)
    root.mainloop()
    
def Begin(imported = False, imgimport = None, refimgimport = None):#begin contains program logic
    global oldclick
    global useold
    global norr
    #global useold
    fontSize = 10;
    plt.rcParams.update({'font.size': fontSize})
    if imported == False:
        raw_path = os.path.normpath(file)
        Image = plt.imread(raw_path)#imfe.get())#filenames[0])#'D:\\Users\\Welcome\\Downloads\\RingPictures.jpg')
        refraw_path = os.path.normpath(reffile)
        refImage = plt.imread(refraw_path)#imfe.get())#filenames[0])#'D:\\Users\\Welcome\\Downloads\\RingPictures.jpg')
    else:
        Image = imgimport#1
        refImage = refimgimport
    
    h, w, _ = Image.shape
    imagepixelwidth = w
    global pixelsze
    global imgDiste
    pixelsize = float(pixelsze.get())#int(input('Please enter the pixelsize: '))
    imagedistance = float(imgDiste.get())#int(input('Please enter the distance in m: '))
    anglePixel = np.arctan(pixelsize/imagedistance)*180/np.pi#obtain angle from here
    #print(anglePixel)
    #print("this is an angle")


    reffig = plt.figure(figsize=(20,20));

    refaxc = reffig.add_subplot(2,2,1)

    refaxc.imshow(refImage);
    refaxc.set_title('reference image');

    # fig2=plt.figure(2)
    refaxc.text(-0.1,-0.1,"click in the center of the diffraction pattern")
    #plt.pause(2)
    


    if useold.get():
        x = oldclick[0]
        y = oldclick[1]
        plt.close(reffig)
    else:
        xy= reffig.ginput(1, show_clicks=True);
        plt.close(reffig)
        x=xy[0][0]
        y=xy[0][1]


    grayImage = rgb2gray(Image)*255;

    fig = plt.figure(figsize=(20,20));

    axc = fig.add_subplot(2,2,1)

    axc.imshow(grayImage,cmap='gray');
    axc.set_title('Original Grayscale Image');

    # fig2=plt.figure(2)
    #axc.text(-0.1,-0.1,"click in the center of the diffraction pattern")
    #plt.pause(2)
    # plt.close(fig2)
    
    oldclick = (x, y)
    print("useold = " + str(useold.get()))
    print("oldclick = " + str(oldclick))
    #print(xy)

    xCenter = x;
    yCenter = y;

    conemultiplier = 5232 / imagepixelwidth
    
    N = 2600 / conemultiplier#1600 for 5232 by 3488 pixels
    N = int(N)
    intensity=np.zeros(N)
    #global ymulte
    intensity2=np.zeros(N)
    
    redmultiplierx = 1
    #bluemultiplierx = 1


    
    #if x > imagepixelwidth / 2:
    #    #clicked to the right
    #    #blue smaller red bigger
    #    tempv = (x - (imagepixelwidth / 2)) / imagepixelwidth #always less than 1
    #    #bluemultiplierx = 0.75 - tempv
    #    redmultiplierx = 1.5 - tempv
    #elif x <= imagepixelwidth / 2:
        #clicked to the left
        #blue bigger red smaller
    tempv = x / imagepixelwidth #always less than 1
        #bluemultiplierx = 1.75 - tempv
    redmultiplierx = tempv * 2#0.95 - tempv

    #print(redmultiplierx)
    #print(bluemultiplierx)
    #print(x)
    #print(imagepixelwidth)
    for radius in tqdm(range(1,N +1)):
    # 	print(radius)

            theta = np.deg2rad(np.linspace(-20, 20, 10)%360);
            theta2 = np.deg2rad(np.linspace(150, 210 , 10));
            #x1= radius * bluemultiplierx * np.cos(theta) + xCenter; 
            #y1 = radius * np.sin(theta) + yCenter; 
            x2= radius * redmultiplierx * np.cos(theta2) + xCenter; 
            y2 = radius * np.sin(theta2)+ yCenter;
            #print(x1)

            #axc.plot(x1, y1, 'b-', linewidth=1);
            axc.plot(x2, y2, 'r-', linewidth=1); 

            #profile=np.zeros(len(x1)) 

            #for k in range(len(x1)):
            #        profile[k] = grayImage[int(round(y1[k])), int(round(x1[k]))]; #logical indexing

            #intensity[radius-1]=np.mean(profile);  

            profile2=np.zeros(len(x2)) 

            for z in range(len(x2)):
                    profile2[z] = grayImage[int(round(y2[z])), int(round(x2[z]))]; # logical indexing

            intensity2[radius-1]=np.mean(profile2);



##    angleXaxis1 = np.linspace(0,len(intensity)*anglePixel,len(intensity));
##    #print("is this the x?")
##    #print(len(intensity)*anglePixel)
##
    axc = fig.add_subplot(6,2,2)
##
##    axc.plot(angleXaxis1,intensity, 'b-')
##    axc.grid(True)
##    axc.set_title('Intensity profile blue');
##    axc.set_xlabel('Angle [degrees]');
##    axc.set_ylabel('Intensity');
##
##    axc = fig.add_subplot(6,2,4)



            
    #def round_to_1(x):
    #    return round(x, -int(floor(log10(abs(x)))))
    
    
    #intensity2 = intensity2 * float(ymulte.get())
    print(intensity2)
    intensity2 = intensity2.tolist()
    #for i in range(len(intensity2)):
        #for x in range(len(intensity2[i])):
    #    intensity2[i] = "{:.5f}".format(float(str(intensity2[i])))
    
    print(intensity2)
    #print(intensity2.tolist())
    print(len(intensity2))
    #intensity2 = map(Decimal,intensity2)
    #print(list(intensity2))
    #intensity2 = list(intensity2)
    #print(intensity2)
    #intensity2 = intensity2 * float(ymulte.get())
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
    with open("Mieplot_Parameters.txt", "a") as f:
        f.write("""
    """)
        f.write("with the save name of :" + str(enam.get()) + " at " + str(time.localtime(time.time())) + "these parameters were used :"
                +"Reference value of : " + str(refve.get())
                +"Refractive index of : " + str(refrve.get())
                +"Lambda value of : " + str(lame.get())
                +"Sigma value of : " + str(sige.get())
                +"Mu value of : " + str(mue.get())
                +"Normal distribution of (1 = normal) : " + str(norr.get())
                +"Amount of Distribution clc points : " + str(distse.get())
                +"Angle clc points of : " + str(thete.get())

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
    #global sdege
    #global edege
    m = ref_Index-0j
    lambda0 = wafeForm  # m = 632e-9#below was originally intensity1
    theta = np.linspace(0,len(intensity2)*anglePixel, theta_var)#int(edege.get()),theta_var)#(int(sdege.get()),int(edege.get()),theta_var)############################################## previously 0 and 10 and theta_var
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
    #print(theta)
    #print("is this the x value you're looking for?")
    ax[0].semilogy(theta, moving_average(sigma_sca, 1)*1e-3, color='tab:orange',
                     label="mean:%.0fnm\n(x10$^{-3}$)" % (np.mean(d)*1e9))
    ax[0].legend()
    ax[0].set_title("Refractive index m=" + str(refrve.get()) + ", $\lambda$=" + str(lame.get()) + "nm & Intensity profile blue")
    ax[0].set_xlabel("Scattering Angle (degrees)")
    ax[0].set_ylabel("Diff. Scattering Cross Section (cm$^2$/sr)")
    ax[0].grid(True)

    
    #X_=np.linspace(x.min(), x.max(), 500)
    #Y_=cubic_interploation_model(X_)
    
    ax2 = ax[0].twinx()  # instantiate a second axes that shares the same x-axis
    angleXaxis2_temp = np.linspace(theta[0],theta[-1],len(angleXaxis2));
    #cubic_interploation_model = interp1d(angleXaxis2_temp, intensity2, kind = "cubic")

    #X_Y_Spline = make_interp_spline(angleXaxis2_temp, intensity2)
    #X_ = np.linspace(angleXaxis2_temp.min(), angleXaxis2_temp.max(), 500)
    #Y_ = X_Y_Spline(X_)
    #ax2.plot(X_, Y_, 'r-', label='Intensity profile blue')#angleXaxis2_temp, intensity2, 'r-', label='Intensity profile blue')
    global ymulte
    global xmulte
    global usecurved
    global userough
    for i in range(len(intensity2)):
        intensity2[i] = intensity2[i] + int(ymulte.get())
    def smooth(y, box_pts):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth
    if usecurved.get():
        ax2.plot(angleXaxis2_temp, smooth(intensity2,int(xmulte.get())), 'r-', label='Intensity profile red')#intensity2, 'r-', label='Intensity profile blue')
        ax2.set_ylabel('Intensity');
        ax2.legend()
        
    if userough.get():
        ax2.plot(angleXaxis2_temp, intensity2,20, 'g-', label='Intensity profile green')#intensity2, 'r-', label='Intensity profile blue')
        ax2.set_ylabel('Intensity');
        ax2.legend()
    

    #shift.yaxis(ax2, float(ymulte.get()), 0.99, False)
    #shift.yaxis(ax2, float(xmulte.get()), 0.5, True)

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
    plt.tight_layout()
    #plt.ylim([0, 250])
    ax2.autoscale(enable=True, axis='y', tight=None)
    fig.tight_layout()
    #ax2.tight_layout()
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
            self.addImgBtn = Button(self.ctrPanel, text="Upload an image", command=self.addrefImage)
            self.addImgBtn.grid(row=0,column=2, pady = 5, sticky =NE)
            self.saveBtn = Button(self.ctrPanel, text="Save", command=self.saveImage)
            self.saveBtn.grid(row=1,column=2, pady = 5, sticky =NE)
            self.saveBtnas = Button(self.ctrPanel, text="Save as", command=self.saveimageas)
            self.saveBtnas.grid(row=2,column=2, pady = 5, sticky =NE)
            self.portBtn = Button(self.ctrPanel, text="Use this image in Mieplot App", command=self.porting)######remember to check back on port
            self.portBtn.grid(row=3,column=2, pady = 5, sticky =NE)
            self.nextBtn = Button(self.ctrPanel, text="Next", command=self.nextimg)
            self.nextBtn.grid(row=4,column=2, pady = 5, sticky =NE)
            self.prevBtn = Button(self.ctrPanel, text="Previous", command=self.previmg)
            self.prevBtn.grid(row=5,column=2, pady = 5, sticky =NE)
            self.canvas.bind('<Configure>', self._resize_image)
            self.croppedref = None



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
                width_factor = self.imgtemp.size[0] / self.sizedimgtemp.size[0]
                height_factor = self.imgtemp.size[1] / self.sizedimgtemp.size[1]#get resized change as a coefficient
                for i in range(4):
                    self.coord[i] = [int(self.coord[i][0] * width_factor), int(self.coord[i][1] * height_factor)]
                    self.startarea = True
                self.Transformer(self.reffile, "reference")
                self.croppedref = self.result_cv
                #print(self.croppedpath)
                cv2.imwrite(self.croppedpath, self.croppedref)
                self.Transformer(self.files[0], self.filenames[0])
                #self.iter += 1###############################################################recently changed
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
            #print(self.iter)
            #print(self.coord)
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
         
            #print(self.coord)
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
        def saveimageas(self):
            ff = [('JPEG', '*.jpg'),
                     ('PNG', '*.png')]
            file = asksaveasfile(filetypes = ff, defaultextension = ff)
            abs_path = os.path.abspath(file.name)
            #self.result_cv.save(abs_path) # saves the image to the input file name.
            file.close()
            os.remove(abs_path)
            if self.startarea == True:
                self.croppedpath = abs_path
            else:
                #cv2.imwrite("result/"+abs_path.split("\\")[-1]+".jpg", self.result_cv)
                cv2.imwrite(abs_path+".jpg", self.result_cv)
            
        def porting(self):
            gotodataapp(True, imgimport=self.result_cv, refimgimport=self.croppedref)#############change to cropped version
        def nextimg(self):
            if self.iter < len(self.files) - 1:
                #self.canvas.delete("all")
                self.iter += 1
                self.Transformer(self.files[self.iter], self.filenames[self.iter])
                
                self.canvas.delete("all")
                self.canvas.create_image(0,0,image=self.result,anchor="nw")
                self.canvas.image = self.result
            elif self.iter >= len(self.files) - 1:
                self.iter = 0
                self.Transformer(self.files[self.iter], self.filenames[self.iter])
                #self.iter += 1
                self.canvas.delete("all")
                self.canvas.create_image(0,0,image=self.result,anchor="nw")
                self.canvas.image = self.result
                
        def previmg(self):
            if self.iter <= 0:
                self.iter = len(self.files) - 1
                self.Transformer(self.files[self.iter], self.filenames[self.iter])
                self.canvas.delete("all")
                self.canvas.create_image(0,0,image=self.result,anchor="nw")
                self.canvas.image = self.result
            elif self.iter < len(self.files):
                #self.canvas.delete("all")
                self.iter -= 1
                self.Transformer(self.files[self.iter], self.filenames[self.iter])
                
                self.canvas.delete("all")
                self.canvas.create_image(0,0,image=self.result,anchor="nw")
                self.canvas.image = self.result

    #---------------------------------
    if __name__ == '__main__':
        
        qroot = Tk()
        #root.geometry("1360x740")
        transformer = PerspectiveTransform(qroot)
        qroot.mainloop()

global info
global numcount
info = [
"""CAMERA PARAMETERS

Light: -1.3
Flash: Off
Background: Third stripe from under
Movement: 7th-8th stripe
Brightness: 20%""",#just add info pages here like these are set out, no additional set up required
"""INSTRUCTIONS PERSPECTIVE TRANSFORMATION APP

Step 1: Enter the amount of images you want to straighten out
Step 2: Choose a reference image (markers should be visible)
Step 3: Choose the dark images (so if you entered e.g. 2 in the textbox you will have to select 2 images
Step 4: If you want to save the transformed reference image, do so before Step 5 with the Save or Save as buttons
Step 5: Click on the markers of the reference image in this order: Top Left, Top Right, Down Left, 
Down Right (DON'T SCROLL ON THIS WINDOW IT WILL MESS UP YOUR CLICKS!)
Step 6: Go through your straightened output images with the next and previous buttons
Step 7: Save the images you need with the 'save' or 'save as' buttons
Step 8: Import your current image with the 'Import' button to the mieplot app""",
"""INSTRUCTIONS MIEPLOT APP
        
Step 1: Fill in all the parameters with your test values
Step 2: Choose a name for your current paramters to be saved with
Step 3: If you didn't import an image from the perspective transformation app then upload an image from your file browser
Step 3: Keep 'Use previous centre' checkbox unticked if this is your first test. If you want test again you can tick it
Step 4: Press the 'start' button.
Step 5: Click on the centre of the ring pattern and wait for the results to be shown
Step 6: Save your results if you need them""",
"""Credits
        
Founder, creator and main developer: Abad Ur-Rehmen Sethi
Co-founder and co-creator: Pieter Verding
""",
"""Update coming soon"""]

numcount = 0
def showinfo():
    global numcount
    global info
    iroot = tk.Tk()
    iroot.geometry("600x300")
    numcount = 0
    global infolabel
    infolabel = Label(iroot, text=info[0])
    infolabel.place(relx=0.5, rely=0.3, anchor=CENTER)
    def ntext():
        global infolabel
        global numcount
        global info
        if numcount < len(info) - 1:
            numcount += 1
            infolabel.config(text=info[numcount])
    def ltext():
        global infolabel
        global numcount
        global info
        if numcount > 0:
            numcount -= 1
            infolabel.config(text=info[numcount])
    
    nextBtn_ = tk.Button(iroot, text="next", command=ntext).place(relx=0.7, rely=0.8, anchor=CENTER)
    PrevBtn_ = tk.Button(iroot, text="prev", command=ltext).place(relx=0.3, rely=0.8, anchor=CENTER)

    
    iroot.mainloop()
    


mroot = tk.Tk()
mroot.resizable(False, False) 
mroot.title("Abad Sethi's Mieplot App")
mroot.geometry("250x150")

sizeapp = tk.Button(mroot, text="Perspective Correction", command=gotosizeapp).place(relx=0.5, rely=0.3, anchor=CENTER)
dataapp = tk.Button(mroot, text="Mieplot App", command=gotodataapp).place(relx=0.5, rely=0.5, anchor=CENTER)
infoBtn = tk.Button(mroot, text="info", command=showinfo).place(relx=0.5, rely=0.7, anchor=CENTER)



mroot.mainloop()
