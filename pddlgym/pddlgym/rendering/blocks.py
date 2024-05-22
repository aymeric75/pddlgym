from .utils import fig2data

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import random

def get_objects_from_obs(obs):
    on_links = {}
    pile_bottoms = set()
    all_objs = set()
    holding = None
    
    for lit in obs:
        if lit.predicate.name.lower() == "ontable":

            pile_bottoms.add(lit.variables[0])
            all_objs.add(lit.variables[0])
        elif lit.predicate.name.lower() == "on":

            on_links[lit.variables[1]] = lit.variables[0]
            all_objs.update(lit.variables)
        elif lit.predicate.name.lower() == "holding":

            
            holding = lit.variables[0]
            all_objs.add(lit.variables[0])
    all_objs = sorted(all_objs)

    bottom_to_pile = {}
    for obj in pile_bottoms:
        bottom_to_pile[obj] = [obj]
        key = obj
        while key in on_links:
            assert on_links[key] not in bottom_to_pile[obj]
            bottom_to_pile[obj].append(on_links[key])
            key = on_links[key]

    piles = []
    for pile_base in all_objs:
        if pile_base in bottom_to_pile:
            piles.append(bottom_to_pile[pile_base])
        else:
            piles.append([])

    return piles, holding


# def get_block_params(piles, width, height, table_height, robot_height):
#     num_blocks = len(piles)
#     #horizontal_padding = 0.025 * width
#     horizontal_padding = 0.08 * width
#     block_width = width / num_blocks - 2*horizontal_padding
#     block_width = 0.04
#     #exit()
#     #block_width = 6
#     #block_height = (height - table_height - robot_height) / num_blocks - 0.05 * height
#     block_height = block_width

#     block_positions = {}
#     for pile_i, pile in enumerate(piles):
#         x = horizontal_padding + pile_i * (block_width + 2*horizontal_padding)
#         for block_i, name in enumerate(pile):
#             y = table_height + block_i * block_height
#             block_positions[name] = (x, y)

#     return block_width, block_height, block_positions


def get_block_params(piles, width, height, table_height, robot_height):
    num_blocks = len(piles)
    horizontal_padding = 0.025 * width
    block_width = width / num_blocks - 2*horizontal_padding
    block_height = (height - table_height - robot_height) / num_blocks - 0.05 * height

    block_positions = {}
    for pile_i, pile in enumerate(piles):
        x = horizontal_padding + pile_i * (block_width + 2*horizontal_padding)
        for block_i, name in enumerate(pile):
            y = table_height + block_i * block_height
            block_positions[name] = (x, y)

    return block_width, block_height, block_positions


def draw_table(ax, width, table_height):
    rect = patches.Rectangle((0,0), width, table_height, 
        linewidth=0, edgecolor=(0.2,0.2,0.2), facecolor=(0.5,0.2,0.0))
    ax.add_patch(rect)

def draw_robot(ax, robot_width, robot_height, midx, midy, holding, block_width, block_height, colors=False):
    x = midx - robot_width/2
    y = midy - robot_height/2
    # rect = patches.Rectangle((x,y), robot_width, robot_height, 
    #     linewidth=1, edgecolor=(0.2,0.2,0.2), facecolor=(0.4, 0.4, 0.4))
    rect = patches.Rectangle((x,y), robot_width, robot_height, 
        linewidth=0, edgecolor=(0.2,0.2,0.2), facecolor=(0.5,0.5,0.5))
    ax.add_patch(rect)


    label=""
    if not colors:
        if str(holding) == 'a:block':
            label="A"
        elif str(holding) == 'b:block':
            label="B"
        elif str(holding) == 'c:block':
            label="C"
        elif str(holding) == 'c:block':
            label="C"
        elif str(holding) == 'd:block':
            label="E"

    # Holding
    if holding is None:
        holding_color = (0.5,0.5,0.5) #(1., 1., 1.)
        ec = (0., 0., 0., 0.)
    else:
        holding_color = block_name_to_color(holding)
        ec = (0.2,0.2,0.2)
    holding_x = midx - block_width/2
    #holding_y = y - block_height/3
    holding_y = y - block_height/2

    # rect = patches.Rectangle((holding_x,holding_y), block_width, block_height, 
    #     linewidth=0, edgecolor=ec, facecolor=holding_color)
    if holding is not None:
        if not colors:
            rect = patches.Rectangle((holding_x,holding_y), block_width, block_height, 
                linewidth=0, edgecolor=ec, facecolor=(0,0,0))
        else:
            rect = patches.Rectangle((holding_x,holding_y), block_width, block_height, 
                linewidth=0, edgecolor=ec, facecolor=holding_color) 
            
    if holding is not None:
        ax.annotate(label, xy=(holding_x+block_width/2, holding_y+block_height/2), weight='bold', color=(1,1,1), fontsize=40, verticalalignment="center", horizontalalignment="center")

    ax.add_patch(rect)

_block_name_to_color = {}
_rng = np.random.RandomState(0)
def block_name_to_color(block_name):
    # colors = [(1, 1, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]

    if block_name not in _block_name_to_color:
        if len(_block_name_to_color) == 0:
            best_color = (0.9, 0.1, 0.1)
        else:
            # Generate 20 random colors and keep the one most different from prior colors
            best_color = None
            max_min_color_diff = 0.
            for _ in range(20):
                color = _rng.uniform(0., 1., size=3)
                min_color_diff = np.inf
                for existing_color in _block_name_to_color.values():
                    diff = np.sum(np.subtract(color, existing_color)**2)
                    min_color_diff = min(diff, min_color_diff)
                if min_color_diff > max_min_color_diff:
                    best_color = color
                    max_min_color_diff = min_color_diff
        _block_name_to_color[block_name] = best_color
    __block_name_to_color = {}
    __block_name_to_color["a:block"] = (1, 1, 0)
    __block_name_to_color["b:block"] = (1, 0, 0)
    __block_name_to_color["c:block"] = (0, 1, 0)
    __block_name_to_color["d:block"] = (0, 0, 1)
    # __block_name_to_color = {
    #     "a:block" : (1, 1, 0),
    #     "b:block" : (1, 0, 0),
    #     "c:block" : (0, 1, 0),
    #     "d:block" : (0, 0, 1)
    # }
    return __block_name_to_color[str(block_name)]

def draw_blocks(ax, block_width, block_height, block_positions, block4=True, colors=False):

    # for block_name, (x, y) in block_positions.items():

    #     print("block_name {}".format(str(block_name)))

    #     label=""

    #     if not colors:

    #         if str(block_name) == 'a:block':
    #             label="A"
    #         elif str(block_name) == 'b:block':
    #             label="B"
    #         elif str(block_name) == 'c:block':
    #             label="C"
    #         elif str(block_name) == 'd:block':
    #             label="E"
    #         else:
    #             label="ok"

    #     color = block_name_to_color(block_name)
    #     if colors:
    #         rect = patches.Rectangle((x,y), block_width, block_height, 
    #             linewidth=0, edgecolor=(0.2,0.2,0.2), facecolor=(0, 0, 0))
    #         print("color")
    #         print(color)
    #         print("rec")
    #         print(rect)
    #     else:
    #         #print("oups")   
    #         rect = patches.Rectangle((x,y), block_width, block_height, 
    #             linewidth=0, edgecolor=(0.2,0.2,0.2), facecolor=(0,0,0))
    #     ax.add_patch(rect)
        
    #     if block4:
    #         ax.annotate(label, xy=(x+block_width/2, y+block_height/2), color=(1,1,1), weight='bold', fontsize=35, verticalalignment="center", horizontalalignment="center")
    #     # else:
    #     #     ax.annotate(label, xy=(x+block_width/2, y+block_height/2), color=(1,1,1), weight='bold', fontsize=40, verticalalignment="center", horizontalalignment="center")


    for block_name, (x, y) in block_positions.items():
        color = block_name_to_color(block_name)
        rect = patches.Rectangle((x,y), block_width, block_height, 
            linewidth=0, edgecolor=(0.2,0.2,0.2), facecolor=color)
        ax.add_patch(rect)


def render(obs, mode='human', close=False, action_was=None):
    width, height = 3.2, 3.2
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0),
                                aspect='equal', frameon=False,
                                xlim=(-0.05, width + 0.05),
                                ylim=(-0.05, height + 0.05))
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(plt.NullFormatter())
        axis.set_major_locator(plt.NullLocator())

    #table_height = height * 0.15
    table_height = height * 0.
    robot_height = height * 0.1
    #robot_height = height * 0.

    piles, holding = get_objects_from_obs(obs)
    
    
    # # if we know the action that led to the current obs
    # if action_was is not None:
    #     # if this action was putdown
    #     if "putdown" in str(action_was):

    #         # retrieve the name of the block that was put down
    #         blockname = str(action_was).split(":")[0][-1] 

    #         indices_to_take_from = []
    #         index_where_putted_block_is=None
    #         putted_ele=None
    #         # 1) retrieve the index of where the block is and retrieve the indices of free spots
    #         for ind, what in enumerate(piles):
    #             if type(what) is list:
    #                 #if a free spots, we add the index
    #                 if len(what) == 0:
    #                     indices_to_take_from.append(ind)
    #                 elif len(what) == 1:
    #                     # if we are at where the putted block is, we retrieve the index
    #                     if what[0].name == blockname:
    #                         indices_to_take_from.append(ind)
    #                         index_where_putted_block_is=ind
    #                         putted_ele=what[0]


    #         # 2) take one of the indices randomly and affect the block to it
    #         theindex = random.choice(indices_to_take_from)
    #         if index_where_putted_block_is != theindex:
    #             piles[theindex] = [putted_ele]
    #             piles[index_where_putted_block_is] = []

    #         # print("THE PILE AFTER")
    #         # print(piles) # 
          

    block_width, block_height, block_positions = get_block_params(piles, width, height, 
        table_height, robot_height)

    robot_width = block_width * 1.4
    robot_midx = width / 2
    robot_midy = height - robot_height/2

    draw_table(ax, width, table_height)
    draw_blocks(ax, block_width, block_height, block_positions)
    draw_robot(ax, robot_width, robot_height, robot_midx, robot_midy, holding,
        block_width, block_height)

    plt.close()


    # print(piles) # [[a:block], [b:block], [c:block]]
    # print(holding) # 



    return fig2data(fig), (piles, holding)





def render3(obs, mode='human', close=False, action_was=None):
    print("renderrrr3")
    width, height = 3.2, 3.2
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0),
                                aspect='equal', frameon=False,
                                xlim=(-0.05, width + 0.05),
                                ylim=(-0.05, height + 0.05))
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(plt.NullFormatter())
        axis.set_major_locator(plt.NullLocator())

    #table_height = height * 0.15
    table_height = height * 0.
    robot_height = height * 0.1
    #robot_height = height * 0.

    piles, holding = get_objects_from_obs(obs)
    
    

    block_width, block_height, block_positions = get_block_params(piles, width, height, 
        table_height, robot_height)

    robot_width = block_width * 1.4
    robot_midx = width / 2
    robot_midy = height - robot_height/2

    draw_table(ax, width, table_height)
    draw_blocks(ax, block_width, block_height, block_positions)
    draw_robot(ax, robot_width, robot_height, robot_midx, robot_midy, holding,
        block_width, block_height)

    plt.close()


    # print(piles) # [[a:block], [b:block], [c:block]]
    # print(holding) # 



    return fig2data(fig), (piles, holding)




def render4(obs, mode='human', close=False, action_was=None):
    print("renderrrr4")
    width, height = 3.2, 3.2
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0),
                                aspect='equal', frameon=False,
                                xlim=(-0.05, width + 0.05),
                                ylim=(-0.05, height + 0.05))
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(plt.NullFormatter())
        axis.set_major_locator(plt.NullLocator())

    #table_height = height * 0.15
    table_height = height * 0.
    robot_height = height * 0.1
    #robot_height = height * 0.

    piles, holding = get_objects_from_obs(obs)
    

    
    block_width, block_height, block_positions = get_block_params(piles, width, height, 
        table_height, robot_height)

    robot_width = block_width * 1.4
    robot_midx = width / 2
    robot_midy = height - robot_height/2

    draw_table(ax, width, table_height)
    draw_blocks(ax, block_width, block_height, block_positions)
    draw_robot(ax, robot_width, robot_height, robot_midx, robot_midy, holding,
        block_width, block_height)

    plt.close()


    # print(piles) # [[a:block], [b:block], [c:block]]
    # print(holding) # 



    return fig2data(fig), (piles, holding)




def render4colors(obs, mode='human', close=False, action_was=None):

    width, height = 3.2, 3.2
    #width, height = 0.2, 0.2
    fig = plt.figure(figsize=(width, height))
    #plt.tight_layout(pad=0)
    # ax = fig.add_axes((0.0, 0.0, 1.0, 1.0),
    #                             aspect='equal', frameon=False,
    #                             xlim=(-0.05, width + 0.05),
    #                             ylim=(-0.05, height + 0.05))
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0),
                                aspect='equal', frameon=False,
                                xlim=(0, width),
                                ylim=(0, height))


    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(plt.NullFormatter())
        axis.set_major_locator(plt.NullLocator())

    #table_height = height * 0.15
    table_height = height * 0.
    robot_height = height * 0.1
    #robot_height = height * 0.


    piles, holding = get_objects_from_obs(obs)

    # print("zzz111111")
    # print(piles)
    # print(holding)
    # exit()
    # for li in piles:
    #     if len(li) > 3:

    #         print(type(li[0]))
    #         print(li[0].name)
        
    #         if li[0].name == "d" and li[1].name == "a" and li[2].name == "b" and li[3].name == "c":
    #             print("lala")
    #             print(li)
    #             exit()


    # print(piles)

    # print(type(holding))


    

    block_width, block_height, block_positions = get_block_params(piles, width, height, 
        table_height, robot_height)

    # print("zzzzzz222222")
    # print(block_width)
    # print(block_height)
    # print(block_positions)
    # exit()

    robot_width = block_width * 1.4
    robot_midx = width / 2
    robot_midy = height - robot_height/2

    draw_table(ax, width, table_height)
    draw_blocks(ax, block_width, block_height, block_positions, colors=True)
    draw_robot(ax, robot_width, robot_height, robot_midx, robot_midy, holding,
        block_width, block_height, colors=True)

    plt.close()


    # print(piles) # [[a:block], [b:block], [c:block]]
    # print(holding) # 



    return fig2data(fig), (piles, holding)
