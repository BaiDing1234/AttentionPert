import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

for see in ['0/2', '1/2', '2/2']:
    see_in_path = ''.join(see.split('/'))
    # Colors for different splits
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'black']
    names_dict = {
            'Ctrl': 'Ctrl',
            'GEARS': f'GEARS',
            'GEARS-g2v': f'g2v',
            'reduce': f'rd',
            'reduce-nocg': f'rd-nocg',
            'reduce-g2v': f'rd-g2v',
            'reduce-nocg-g2v': f'rd-nocg-g2v',
            'only bp': f'only PL',
            'only pw': f'only PW',
            'Ours-g2v': f'Ours',
        }

    # Creating the plot
    plt.figure(figsize=(4, 4))
    y_all = []
    

    for i in range(1, 6):
        # Read the CSV file
        file_name = f'csvs/abl_split_{i}_seen.csv'
        data = pd.read_csv(file_name)

        # Assuming 'model' column contains categorical data
        x_values = [names_dict[s] for s in data['model'].astype(str)][1:]
        
        y_values = np.array([float(s.split('_')[0].strip('$')) for s in data[f'MSE Seen {see} Percent']])[1:]
        y_all.append(y_values)

        # Plotting the data
        plt.plot(x_values, y_values, '--', color=colors[i-1], label=f'Split {i}')
    y_min = np.array(y_all).min()
    #y_avg = np.array(y_all).mean(0)
    #plt.plot(x_values, y_avg, color=colors[-1], label=f'Average')

    # Customizing the plot
    plt.xlabel('Model')
    #plt.text(-0.1, -0.1, 'Model:', transform=plt.gca().transAxes, fontsize=10, va='top', ha='right')
    plt.ylabel(f'Seen {see} rel-MSE(DE) (%)')
    #bottom = 10 * int(y_min/10)
    plt.ylim(10 , 100)
    #plt.title('MSE Values by Model for Different Splits')
    plt.legend(fontsize=8, loc = 'upper right')
    plt.xticks(rotation=45)  # Rotating the x-axis labels for better readability
    plt.tight_layout()

    # save the plot
    plt.savefig(f'figs/abl_splits_plot_{see_in_path}.pdf')
    #plt.close()
