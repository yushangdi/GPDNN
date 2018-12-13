import matplotlib.pyplot as plt


# \begin{tabular}{lcccr}
# \hline
# \abovespace\belowspace
# samples & ConvNet & ResNet & Densenet & RBF \\
# \hline
# \abovespace
# 1000    & 38.6 & 39.2 & 39.0 & 0\\
# 5000    & 50.9 & 49.8 & 49.7 & 0\\
# 25000    & 61.6 & 56.5 & 60.7 & 0\\
# \midrule
# 1000    & 39 & 39.9 & 40.1 & -\\
# 5000    & 51.9 & 50.6 & 50.9 & -\\
# 25000    & 63.4 & 61.4 & 62 & -\\
# \midrule
# 1000    & 39.2 & 40.6 & 40.9 & -\\
# 5000    & 51.5 & 50.5 & 50.2 & -\\
# 25000    & x & 61.9 & 60.9 & -\\
# \hline
# \end{tabular}

# In[29]:
x = [1000,5000,25000]

convnet = [38.6,50.9,61.6]
resnet = [39.2,49.8,56.5]
densenet = [39.0,49.7,60.7]
plt.plot(x,convnet, label = "convnet")
plt.scatter(x,convnet)
plt.plot(x,resnet, label = "resnet")
plt.scatter(x,resnet)
plt.plot(x,densenet, label = "densenet")
plt.scatter(x,densenet)
plt.xscale('log')
plt.legend(prop={'size': 13})
plt.title("n = 26",fontsize=15)
plt.xticks(x, x)
plt.xlabel("Samples",fontsize=15)
plt.ylabel("GP Accuracy",fontsize=15)
plt.ylim((30, 70))
plt.show()



convnet = [39,51.9,63.4]
resnet = [39.9,50.6,61.4]
densenet = [40.1,50.9,62]
plt.plot(x,convnet, label = "convnet")
plt.scatter(x,convnet)
plt.plot(x,resnet, label = "resnet")
plt.scatter(x,resnet)
plt.plot(x,densenet, label = "densenet")
plt.scatter(x,densenet)
plt.xscale('log')
plt.legend(prop={'size': 13})
plt.title("n = 38")
plt.xticks(x, x)
plt.xlabel("Samples",fontsize=15)
plt.ylabel("GP Accuracy",fontsize=15)
plt.ylim((30, 70))
plt.show()


convnet = [39.2,51.5,63]
resnet = [40.6,50.5,61.9]
densenet = [40.9,50.2,60.9]

plt.plot(x,convnet, label = "convnet")
plt.scatter(x,convnet)
plt.plot(x,resnet, label = "resnet")
plt.scatter(x,resnet)
plt.plot(x,densenet, label = "densenet")
plt.scatter(x,densenet)
plt.xscale('log')
plt.legend(prop={'size': 13})
plt.title("n = 50")
plt.xticks(x, x)
plt.xlabel("Samples",fontsize=15)
plt.ylabel("GP Accuracy",fontsize=15)
plt.ylim((30, 70))
plt.show()

# \abovespace\belowspace
# samples & ConvNet & ResNet & Densenet & RBF \\
# \hline
# \abovespace
# 1000    & 98.13 & 108.47 & 120.57 & 0\\
# 5000    & 519.49 & 390.21 & 407.99 & 0\\
# 25000    & 9316.94 & 2323.73 & 2327.48 & 0\\
# \midrule
# 1000    & 116.5 & 126.77 & 177.95 & -\\
# 5000    & 718.21 & 529.16 & 588.85 & -\\
# 25000    & 13253.51 & 8700 & 8580.55 & -\\
# \midrule
# 1000    & 135.24 & 150.85 & 250.51 & -\\
# 5000    & 920.61 & 675.59 & 794.19 & -\\
# 25000    & x & 11554.01 & 12381.11 & -\\
# \hline

# In[76]:


def plot_table(convnet,resnet,densenet,ylabel, ylim, title ):
    x_lst = np.arange(0,25000,50)
    plt.plot(x,convnet, label = "convnet")
    plt.scatter(x,convnet)
    plt.plot(x,resnet, label = "resnet")
    plt.scatter(x,resnet)
    plt.plot(x,densenet, label = "densenet")
    plt.scatter(x,densenet)
    # plt.xscale('log')
    plt.title(title)
    plt.xticks(x, x)
    plt.xlabel("Samples",fontsize=15)
    plt.ylabel(ylabel,fontsize=15)
    plt.ylim((0, ylim))



# In[79]:


ylabel = "GP Running Time"
ylim = 18000

title = "n = 26"
convnet = [98.13, 519.49,9316.94]
resnet = [108.47,390.21,2323.73 ]
densenet = [120.57,407.99, 2327.48]
plot_table(convnet,resnet,densenet,ylabel, ylim, title )
plt.plot(x_lst,x_lst**2/100000,'--', label = "$y=Cx^2$")
plt.legend(prop={'size': 13},loc=2)
plt.show()

title = "n = 38"
convnet = [116.5,718.21,13253.51]
resnet = [126.77,529.16,8700 ]
densenet = [177.95,588.85, 8580.55]
plot_table(convnet,resnet,densenet,ylabel, ylim, title )
plt.plot(x_lst,x_lst**2/60000,'--', label = "$y=Cx^2$")
plt.legend(prop={'size': 13})
plt.ylim((0,18000))
plt.show()

title = "n = 50"
convnet = [135.24,920.61,17000]
resnet = [150.85,675.59,11554.01]
densenet = [250.51,794.19, 12381.11]
plot_table(convnet,resnet,densenet,ylabel, ylim, title )
plt.plot(x_lst,x_lst**2/40000, label = "$y=Cx^2$")
plt.legend(prop={'size': 13})
plt.ylim((0,18000))
plt.show()

# In[65]:


convnet = np.array([98.13, 519.49,9316.94])**0.5
resnet = np.array([108.47,390.21,2323.73 ])**0.5
densenet = np.array([120.57,407.99, 2327.48])**0.5
plt.plot(x,convnet, label = "convnet")
plt.scatter(x,convnet)
plt.plot(x,resnet, label = "resnet")
plt.scatter(x,resnet)
plt.plot(x,densenet, label = "densenet")
plt.scatter(x,densenet)
# plt.xscale('log')
plt.legend(prop={'size': 13})
plt.title("n = 26")
plt.xticks(x, x)
plt.xlabel("Samples",fontsize=15)
plt.ylabel("GP Running Time",fontsize=15)
plt.show()
convnet = np.array([116.5,718.21,13253.51])**0.5
resnet = np.array([126.77,529.16,8700 ])**0.5
densenet = np.array([177.95,588.85, 8580.55])**0.5
plt.plot(x,convnet, label = "convnet")
plt.scatter(x,convnet)
plt.plot(x,resnet, label = "resnet")
plt.scatter(x,resnet)
plt.plot(x,densenet, label = "densenet")
plt.scatter(x,densenet)
# plt.xscale('log')
plt.legend(prop={'size': 13})
plt.title("n = 38")
plt.xticks(x, x)
plt.xlabel("Samples",fontsize=15)
plt.ylabel("GP Running Time",fontsize=15)
plt.show()
convnet = np.array([135.24,920.61,17000])**0.5
resnet =np.array( [150.85,675.59,11554.01])**0.5
densenet =np.array( [250.51,794.19, 12381.11])**0.5
plt.plot(x,convnet, label = "convnet")
plt.scatter(x,convnet)
plt.plot(x,resnet, label = "resnet")
plt.scatter(x,resnet)
plt.plot(x,densenet, label = "densenet")
plt.scatter(x,densenet)
# plt.xscale('log')
plt.legend(prop={'size': 13})
plt.title("n = 50")
plt.xticks(x, x)
plt.xlabel("Samples",fontsize=15)
plt.ylabel("GP Running Time",fontsize=15)


# In[ ]:
