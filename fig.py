import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D


fig_size = (6,5)

def create_3d_contour():
    def complex_rbf(x, y, z):
        r = np.sqrt(x**2 + y**2 + z**2)
        return np.sin(3*r)/r + 0.1*(np.cos(2*x) + np.sin(2*y) + np.cos(2*z))

    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    z = np.linspace(-5, 5, 100)
    X, Y, Z = np.meshgrid(x, y, z)

    values = complex_rbf(X, Y, Z)

    # Calculate the actual range of values in the XY plane we're visualizing
    xy_plane_values = values[:,:,50]
    z_min, z_max = xy_plane_values.min(), xy_plane_values.max()

    global fig_size
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection='3d')

    # Create multiple contour surfaces
    levels = np.linspace(z_min, z_max, 60)
    for level in levels:
        ax.contour(X[:,:,50], Y[:,:,50], xy_plane_values, levels=[level], colors=['#4287f5'], alpha=0.2)

    ax.set_xlabel('X-axis', labelpad=-10)
    ax.set_ylabel('Y-axis', labelpad=-10)
    ax.set_zlabel('Z-axis', labelpad=-10)
    ax.set_title('3D Contour Plot For RBF', pad=10)

    # Remove all panes, grids, and tick labels
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    # Make panes transparent and set background color to white
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    # Adjust the plot limits to focus on the contour region
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(z_min, z_max)  # Use the actual range of values

    # Adjust the viewing angle for better visibility
    ax.view_init(elev=15, azim=45)

    #ax.view_init(elev=0, azim=45)


    # Create legend
    legend_elements = [Line2D([0], [0], color='#4287f5', lw=2, label='Contours in XY-planes')]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1), fontsize='small')

    # Adjust the layout to minimize white space
    plt.tight_layout(pad=0.1)

    # Adjust the axes position to fill more of the figure
    ax.set_position([0.05, 0.05, 0.9, 0.9])

    # Increase the vertical stretch of the plot
    scale_factor = 1.2
    scale_matrix = np.diag([1, 1, scale_factor, 1])
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), scale_matrix)

    plt.savefig('RBF-contour.pdf', bbox_inches='tight', pad_inches=0.1)

    plt.show()


create_3d_contour()