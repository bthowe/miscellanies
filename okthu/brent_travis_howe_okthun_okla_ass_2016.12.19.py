import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import percentileofscore
from sklearn.grid_search import GridSearchCV
from sklearn.metrics.scorer import make_scorer
from sklearn.cross_validation import train_test_split
from matplotlib.patches import Circle, Rectangle, Arc
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import ElasticNet, Ridge, Lasso

def data():
    df = pd.read_csv('../hackerrank_data/shots.csv')
    df = df[(np.sqrt(df.shot_y**2 + df.shot_x**2)<26.75)] # restrict data to "normal" shots (i.e., within 26'9'')

    column_names = df.columns
    df = pd.DataFrame(df.values, columns = column_names) # reset dataframe index


    #feature engineering, define feature and outcome matrix
    df['three_pointer'] = 0
    df.ix[(np.sqrt(df.shot_y**2 + df.shot_x**2)>=23.75) | ((np.abs(df.shot_x)>=22) & (df.shot_y<(14-4-.75))), 'three_pointer'] = 1

    df['degrees'] = np.degrees(np.arccos(((df.shot_x*0 + df.shot_y*1) / np.sqrt(df.shot_x**2 + df.shot_y**2))))
    df['distance'] = np.sqrt(df.shot_x**2 + df.shot_y**2)
    X = df.copy()

    X['shooter_velocity_angle'] = np.abs(X['shooter_velocity_angle'])
    X['defender_angle'] = np.abs(X['defender_angle'])
    X['defender_velocity_angle'] = np.abs(X['defender_velocity_angle'])

    X.drop(['shot_x', 'shot_y'], 1, inplace=True)

    y = X.pop('made')

    return df, X, y


def en(X, y):
    en = ElasticNet(normalize=True)

    param_dict = {'alpha': [0.0000000001, 0.0000005, 0.0000075, 0.000001, 0.000005, 0.0001, 0.001, 0.01, 0.1, .25, .5, .75, 1],
        'l1_ratio': [0, 0.0002, .45, .475, .5, .525, .55, .95, 1]}
    gsCV_en = GridSearchCV(en, param_dict, n_jobs = -1)
    gsCV_en.fit(X, y)

    print gsCV_en.best_params_
    print gsCV_en.best_score_

    return gsCV_en.predict(X)


def boots_r(pred_prob, actual):
    sample_size = 1000

    boot_shooting_perc_dist = []
    for i in xrange(sample_size):
        boot_perc_sample = [pred_prob[ind] for ind in np.random.choice(len(pred_prob), len(pred_prob), replace=True)]
        samp_mean = np.mean(np.where(np.random.uniform(0, 1, len(pred_prob))<=boot_perc_sample, 1, 0))
        boot_shooting_perc_dist.append(samp_mean)
    return percentileofscore(boot_shooting_perc_dist, actual, kind='weak')

def diagram_data(bin_num, min_obs, var, df):
    d = stats.binned_statistic_2d(df.shot_x, df.shot_y, var, statistic='mean', bins=bin_num)
    statistic, xedges, yedges, binnumber = d

    binnumber = pd.Series(binnumber)

    bin_max_num = binnumber.max()

    stat = []
    min_obs = 20
    for i in xrange(len(xedges)-1):
        i_row = []
        for j in xrange(len(yedges)-1):
            bin = df[((df.shot_x>=xedges[i]) & (df.shot_x<xedges[i+1])) & ((df.shot_y>=yedges[j]) & (df.shot_y<yedges[j+1]))]
            l = len(bin)
            if l<min_obs:
                i_row.append(np.nan)
            else:
                b = boots_r(var[bin.index.values], bin.made.mean())
                if b>=95:
                    i_row.append(1)
                elif b<=5:
                    i_row.append(-1)
                else:
                    i_row.append(0)
        stat.append(i_row)
    return np.array(stat), d


# I, conveniently, found this (modulo a few modifications) online
def draw_court(ax=None, color='black', lw=2, outer_lines=False):
    # If an axes object isn't provided to plot onto, just get current one
    if ax is None:
        ax = plt.gca()

    # Create the various parts of an NBA basketball court

    # Create the basketball hoop
    # Diameter of a hoop is 18" so it has a radius of 9", which is a value
    # 7.5 in our coordinate system
#     hoop = Circle((0, 0), radius=7.5, linewidth=lw, color=color, fill=False)
    hoop = Circle((0, 0), radius=.75, linewidth=lw, color=color, fill=False)

    # Create backboard
    backboard = Rectangle((-3, -.75), 6, -0.125, linewidth=lw, color=color)

    # The paint
    # Create the outer box 0f the paint, width=16ft, height=19ft
    outer_box = Rectangle((-8, -4), 16, 19, linewidth=lw, color=color,
                          fill=False)
    # Create the inner box of the paint, widt=12ft, height=19ft
    inner_box = Rectangle((-6, -4), 12, 19, linewidth=lw, color=color,
                          fill=False)

    # Create free throw top arc
    top_free_throw = Arc((0, 14.75), 12, 12, theta1=0, theta2=180,
                         linewidth=lw, color=color, fill=False)
    # Create free throw bottom arc
    bottom_free_throw = Arc((0, 14.75), 12, 12, theta1=180, theta2=0,
                            linewidth=lw, color=color, linestyle='dashed')
    # Restricted Zone, it is an arc with 4ft radius from center of the hoop
    restricted = Arc((0, 0), 8, 8, theta1=0, theta2=180, linewidth=lw,
                     color=color)

    # Three point line
    # Create the side 3pt lines, they are 14ft long before they begin to arc
    corner_three_a = Rectangle((-22, -4), 0, 13.75, linewidth=lw,
                               color=color)
    corner_three_b = Rectangle((22, -4), 0, 13.75, linewidth=lw, color=color)
    # 3pt arc - center of arc will be the hoop, arc is 23'9" away from hoop
    # I just played around with the theta values until they lined up with the
    # threes
    three_arc = Arc((0, 0), 48, 48, theta1=23.75, theta2=156.25, linewidth=lw,
                    color=color)

    # Center Court
    center_outer_arc = Arc((0, 422.5), 120, 120, theta1=180, theta2=0,
                           linewidth=lw, color=color)
    center_inner_arc = Arc((0, 422.5), 40, 40, theta1=180, theta2=0,
                           linewidth=lw, color=color)

    # List of the court elements to be plotted onto the axes
    court_elements = [hoop, backboard, outer_box, inner_box, top_free_throw,
                      bottom_free_throw, restricted, corner_three_a,
                      corner_three_b, three_arc, center_outer_arc,
                      center_inner_arc]

    if outer_lines:
        # Draw the half court line, baseline and side out bound lines
        outer_lines = Rectangle((-25, -4), 50, 47, linewidth=lw,
                                color=color, fill=False)
        court_elements.append(outer_lines)

    # Add the court elements onto the axes
    for element in court_elements:
        ax.add_patch(element)

    return ax



if __name__=="__main__":
    #inital contour plot
    df = pd.read_csv('../hackerrank_data/shots.csv')
    df = df[df.shot_y<=40]
    heatmap, xedges, yedges = np.histogram2d(df.shot_x, df.shot_y, bins=50)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    fig = plt.figure(figsize=(18,18))
    ax = fig.add_subplot(1,1,1)
    ax.imshow(heatmap.T, extent=extent, origin='lower')
    plt.title('Figure 1')
    plt.savefig('heatmap.png')
    plt.show()

    #contour plot of shooting percentage
    bin_num = 20
    min_obs = 30

    d_mean = stats.binned_statistic_2d(df.shot_x, df.shot_y, df.made, statistic='mean', bins=bin_num)
    d_count = stats.binned_statistic_2d(df.shot_x, df.shot_y, df.made, statistic='count', bins=bin_num)
    d = np.array([np.where(np.array(d_count[0])>min_obs, np.array(d_mean[0]), np.nan), np.array(d_mean[1]), np.array(d_mean[2])])

    extent = [d[1][0], d[1][-1], d[2][0], d[2][-1]]
    fig = plt.figure(figsize=(16,16))
    ax = fig.add_subplot(1,1,1)
    im = ax.imshow(d[0].T, cmap='YlOrRd', extent=extent, interpolation='nearest', origin='lower') # RdBu_r

    plt.title('Figure 2')
    cbar_ax = fig.add_axes([0.85, 0.65, 0.05, 0.2])
    cb = plt.colorbar(im, cax=cbar_ax)
    cb.set_label('shooting percentage')

    plt.savefig('shooting_percentage.png')
    plt.show()



    #primary analysis
    df, X, y = data()
    y_pred = en(X, y)

    #plot 3
    bin_num = 20
    min_obs = 20
    hey, d = diagram_data(bin_num, min_obs, y_pred, df)

    extent = [d[1][0], d[1][-1], d[2][0], d[2][-1]]
    fig = plt.figure(figsize=(16,14))
    ax = fig.add_subplot(1,1,1)
    draw_court(outer_lines=True)

    im = ax.imshow(hey.T, cmap='YlOrRd', extent=extent, interpolation='nearest', origin='lower') #
    plt.title('Figure 3')
    cbar_ax = fig.add_axes([0.85, 0.65, 0.05, 0.2])
    cb = plt.colorbar(im, cax=cbar_ax)
    cb.set_label('Outlier Index')

    plt.savefig('shooting_percentage1.png')
    plt.show()

    #plot 4
    bin_num = 30
    min_obs = 10
    hey, d = diagram_data(bin_num, min_obs, y_pred, df)

    extent = [d[1][0], d[1][-1], d[2][0], d[2][-1]]
    fig = plt.figure(figsize=(16,14))
    ax = fig.add_subplot(1,1,1)
    draw_court(outer_lines=True)

    im = ax.imshow(hey.T, cmap='YlOrRd', extent=extent, interpolation='nearest', origin='lower') #
    plt.title('Figure 4')
    cbar_ax = fig.add_axes([0.85, 0.65, 0.05, 0.2])
    cb = plt.colorbar(im, cax=cbar_ax)
    cb.set_label('Outlier Index')

    plt.savefig('shooting_percentage2.png')
    plt.show()
