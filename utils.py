import matplotlib.pyplot as plt
import matplotlib.animation as animation

def anim(frames):
    plt.figure(figsize=(frames[0].shape[1]/72.0/4, frames[0].shape[0]/72.0/4), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=1000)
    plt.show()