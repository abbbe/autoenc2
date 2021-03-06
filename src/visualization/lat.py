import matplotlib.pyplot as plt
from matplotlib import animation

from src.visualization.dataset import _clean_ax, _imshow

def fine_scatter(data1, x1, i1, data2, x2, i2):
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.scatter(data1[x1][:,i1], data2[x2][:,i2], 1)
    ax.set_xlabel("%s%d" % (x1, i1))
    ax.set_ylabel("%s%d" % (x2, i2))
    ax.grid()

def fine_scatter_color(data1, x1, i1, data2, x2, i2, c=None, size=10, ax=None):
    if ax is None:
        _fig, ax = plt.subplots(figsize=(size, size))
        
    ax.set_facecolor('xkcd:black')

    x, y = data1[x1][:,i1], data2[x2][:,i2]
    ax.scatter(x, y, 1, c=c)
    ax.set_xlabel("%s%d" % (x1, i1))
    ax.set_ylabel("%s%d" % (x2, i2))
    #ax.grid()

def fine_scatter_sum(data1, x1, i1a, i1b, data2, x2, i2):
    fig, ax = plt.subplots(figsize=(10, 10))

    x = data1[x1][:,i1a] + data1[x1][:,i1b]
    ax.scatter(x, data2[x2][:,i2], 1)
    
    ax.set_xlabel("%s(%d+%d)" % (x1, i1a, i1b))
    ax.set_ylabel("%s%d" % (x2, i2))
    ax.grid()

def fine_scatter_color_sum(data1, x1, i1a, i1b, data2, x2, i2, c=None, size=10, ax=None):
    if ax is None:
        _fig, ax = plt.subplots(figsize=(size, size))
        
    ax.set_facecolor('xkcd:black')

    x = data1[x1][:,i1a] + data1[x1][:,i1b]
    y = data2[x2][:,i2]
    ax.scatter(x, data2[x2][:,i2], 1, c)
    
    ax.set_xlabel("%s(%d+%d)" % (x1, i1a, i1b))
    ax.set_ylabel("%s%d" % (x2, i2))

def fine_scatter_color_sub(data1, x1, i1a, i1b, data2, x2, i2, c=None, size=10, ax=None):
    if ax is None:
        _fig, ax = plt.subplots(figsize=(size, size))
        
    ax.set_facecolor('xkcd:black')

    x = data1[x1][:,i1a] - data1[x1][:,i1b]
    y = data2[x2][:,i2]
    ax.scatter(x, data2[x2][:,i2], 1, c)
    
    ax.set_xlabel("%s(%d-%d)" % (x1, i1a, i1b))
    ax.set_ylabel("%s%d" % (x2, i2))

def display_xvars(data):
    nLat = data['rec_images'].shape[1]
    nVoltages = data['angles'].shape[1]

    fig, axs = plt.subplots(nLat+1, nVoltages+1)
    fig.tight_layout()

    for i in range(nLat):
      for j in range(nVoltages):
        title = "volt%d vs lat%d" % (j, i)
        axs[i][j].title.set_text(title)
        axs[i][j].plot(data['rec_images'][:, i], data['angles'][:,j], '.')

    axs[nLat][0].title.set_text("volt0 vs volt1")
    axs[nLat][0].plot(data['angles'][:,1], data['angles'][:,0], '.')

    axs[0][nVoltages].title.set_text("lat1 vs lat0")
    axs[0][nVoltages].plot(data['rec_images'][:,0], data['rec_images'][:,1], '.')

    _clean_ax(axs[1][2])
    _clean_ax(axs[2][1])
    _clean_ax(axs[2][2])

def visualize_lat_space(dataset_grid, dataset_grid_out, sheet):
    fig, axs = plt.subplots(2, 2, figsize=(15,12))
    fig.tight_layout()

    if sheet == 1:
        fine_scatter_color(dataset_grid, 'angles', 0, dataset_grid_out, 'latvars' , 0, dataset_grid['angles'][:,1], ax=axs[0,0])
        fine_scatter_color(dataset_grid, 'angles', 1, dataset_grid_out, 'latvars' , 1, dataset_grid['angles'][:,0], ax=axs[0,1])

        fine_scatter_color(dataset_grid, 'angles', 0, dataset_grid_out, 'latvars' , 1, dataset_grid['angles'][:,1], ax=axs[1,0])
        fine_scatter_color(dataset_grid, 'angles', 1, dataset_grid_out, 'latvars' , 0, dataset_grid['angles'][:,0], ax=axs[1,1])

    elif sheet == 2:
        fine_scatter_color_sum(dataset_grid, 'angles', 0, 1, dataset_grid_out, 'latvars', 0, dataset_grid_out['latvars'][:,1], ax=axs[0,0])
        fine_scatter_color_sum(dataset_grid, 'angles', 0, 1, dataset_grid_out, 'latvars', 1, dataset_grid_out['latvars'][:,0], ax=axs[0,1])

        fine_scatter_color_sub(dataset_grid, 'angles', 0, 1, dataset_grid_out, 'latvars', 0, dataset_grid_out['latvars'][:,1], ax=axs[1,0])
        fine_scatter_color_sub(dataset_grid, 'angles', 0, 1, dataset_grid_out, 'latvars', 1, dataset_grid_out['latvars'][:,0], ax=axs[1,1])
    else:
        raise
    
    return fig, axs

def animate_Y_YYs(ys, yys, outfile):
    fig, axs = plt.subplots(1, 2, figsize=(4, 2))
    fig.tight_layout()

    _clean_ax(axs[0])
    _clean_ax(axs[1])

    def animate(frame):
        _imshow(axs[0], ys[frame])
        _imshow(axs[1], yys[frame])
        return (fig,)

    N = ys.shape[0]
    assert(N == yys.shape[0])
    ani = animation.FuncAnimation(fig, animate, frames=range(N), blit=True)
    ani.save(outfile)
    fig.clf()
    
    return None