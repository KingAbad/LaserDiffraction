import numpy as np 
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from tqdm import tqdm
  
try:
    import miepython

except ModuleNotFoundError:
    print('miepython not installed. To install, uncomment and run the cell above.')
    print('Once installation is successful, rerun this cell again.')
    
fontSize = 10;
plt.rcParams.update({'font.size': fontSize})



pixelsize = 0.00005 
imagedistance = 0.8 
anglePixel = np.arctan(pixelsize/imagedistance)*180/np.pi
Image = plt.imread(r'C:\Users\drago\Desktop\RingPictures.jpg');
grayImage = rgb2gray(Image)*255;

fig = plt.figure(figsize=(20,20));

axc = fig.add_subplot(2,2,1)

axc.imshow(grayImage,cmap='gray');
axc.set_title('Original Grayscale Image');

fig2=plt.figure(2)
axc.text(-0.1,-0.1,"click in the center of the diffraction pattern")
#plt.pause(2)
plt.close(fig2)

xy= fig.ginput(1);
x=xy[0][0]
y=xy[0][1]


xCenter = x;
yCenter = y;

N=1600
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

	axc.plot(x1, y1, 'b-', linewidth= 1);
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
d_ref = 20000e-9

''' let the user decide'''
# mu_distr = float(input('Please input mu: '))
# sigma_distr = float(input('Please enter sigma: '))
# distr = int(input('Please choose the distribution:\n1:\tNormal\t2:\tLognormal\n'))
# while distr!=1 and distr!=2:
#     distr = int(input('Invalid choice!\nPlease choose the distribution:\n1:\tNormal\t2:\tLognormal\n'))

''' try normal'''
mu_distr, sigma_distr = (d_ref, 10/100*d_ref)
distr = 1
size_distr = int(1e4)

''' try lognormal'''
# mu_distr, sigma_distr = (d_ref, 10/100*d_ref)
# distr = 2
# size_distr = int(1e3)

''' param definitions '''
m = 1.33-0j
lambda0 = 632e-9  # m
theta = np.linspace(0,10,1024)
mu = np.cos(theta* np.pi/180)

''' choose the distribution '''
if distr == 1:
    d_orig = np.random.normal(mu_distr, sigma_distr, size=size_distr) # m; Normal
    bins_h = np.histogram_bin_edges(np.random.normal(mu_distr, sigma_distr/np.sqrt(2*np.pi), size=int(1e6)), size_distr)
    d = bins_h[:-1]
elif distr == 2:
    d_orig = np.random.lognormal(mu_distr, sigma_distr, size=size_distr)  # m; Logormal
    bins_h = np.histogram_bin_edges(np.random.lognormal(mu_distr, sigma_distr/np.sqrt(2*np.pi), size=int(1e6)), size_distr)
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
    # print(idx, '/', size_distr)
#%% plotting part
# fig, ax = plt.subplots(1,2,figsize=(15,5))

axc = fig.add_subplot(2,1,2)

axc.semilogy(theta, sigma_sca_ref*1e-3,
               color='tab:blue', label="%.0fnm\n(x10$^{-3}$)" % (d_ref*1e9))
axc.semilogy(theta, moving_average(sigma_sca, 1)*1e-3, color='tab:orange',
                 label="mean:%.0fnm\n(x10$^{-3}$)" % (np.mean(d)*1e9))
axc.legend()
axc.set_title("Refractive index m=1.4, $\lambda$=532nm")
axc.set_xlabel("Scattering Angle (degrees)")
axc.set_ylabel("Diff. Scattering Cross Section (cm$^2$/sr)")
axc.grid(True)

axc = fig.add_subplot(6,2,6)

count_h, bins_h, _ = axc.hist(d_orig, 30, density=True, label="Histogram,\n$\mu$=%.2g $\sigma$=%.2g" % (np.mean(d_orig), np.std(d_orig)))
if distr == 1:
    # Normal
    pdf_distr = 1/(sigma_distr * np.sqrt(2 * np.pi)) *\
          np.exp( - (bins_h - mu_distr)**2 / (2 * sigma_distr**2))
    title_str = " (Normal Distribution)"
elif distr == 2:
    # Logormal
    pdf_distr = (np.exp(-(np.log(bins_h) - mu_distr)**2 / (2 * sigma_distr**2))/\
              (bins_h * sigma_distr * np.sqrt(2 * np.pi)))
    title_str = " (Lognormal Distribution)"
axc.plot(bins_h, pdf_distr, linewidth=2, color='tab:red',
            label="Theoretical pdf,\n$\mu$=%.2g $\sigma$=%.2g" % (mu_distr, sigma_distr))
axc.set_title("Histogram vs PDF of d" + title_str)
axc.set_xlabel("d (m)")
axc.set_ylabel("Density")
axc.grid(True)
axc.legend()

plt.show()