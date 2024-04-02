import seaborn as sns
import matplotlib.pyplot as plt
import DC_modeling

g = sns.lineplot(x=[],y=[])

for i in range(10):
    x,y = DC_modeling.update()
    print(x,y)
    g.set_data(x,y)
    plt.draw()
    print(i)
    if i == 0:
        plt.show()

