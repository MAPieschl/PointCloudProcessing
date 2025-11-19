import PointCloudSet
import pickle
import sys
import os

if __name__=="__main__":

    aftr_path: str = ""

    try:
        aftr_path = sys.argv[1]

    except:
        print('Please input a path to the Aftr directory')

    if(os.path.isdir(aftr_path)):
        pc = PointCloudSet.PointCloudSet(one_hot = True,
                                class_labels = ['kc46'], 
                                part_labels = ['fuselage', 'left_engine', 'right_engine', 'left_wing', 'right_wing', 'left_hstab', 'right_hstab', 'vstab', 'left_boom_stab', 'right_boom_stab', 'boom_wing', 'boom_hull', 'boom_hose'], 
                                pretrain_tnet = False, 
                                network_input_width = 4096,
                                batch_size = 8,
                                rand_seed = 42)
        pc.build_from_aftr_output(aftr_path)
        pc.get_info()
        with open(f"data/{aftr_path.split('/')[-1]}.pkl", 'wb') as p:
            pickle.dump(pc, p)

    else:
        print('Invalid path provided.')