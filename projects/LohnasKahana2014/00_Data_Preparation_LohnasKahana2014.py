# %% [markdown]
# # Lohnas & Kahana, 2014 Dataset

# %% [markdown]
# > Siegel, L. L., & Kahana, M. J. (2014). A retrieved context account of spacing and repetition effects in free recall. Journal of Experimental Psychology: Learning, Memory, and Cognition, 40(3), 755.
# 
# Across 4 sessions, 35 subjects performed delayed free recall of 48 lists. Subjects were University of Pennsylvania undergraduates, graduates and staff, age 18-32. List items were drawn from a pool of 1638 words taken from the University of South Florida free association norms (Nelson, McEvoy, & Schreiber, 2004; Steyvers, Shiffrin, & Nelson, 2004, available at http://memory.psych.upenn.edu/files/wordpools/PEERS_wordpool.zip). Within each session, words were drawn without replacement. Words could repeat across sessions so long as they did not repeat in two successive sessions. Words were also selected to ensure that no strong semantic associates co-occurred in a given list (i.e., the semantic relatedness between any two words on a given list, as determined using WAS (Steyvers et al., 2004), did not exceed a threshold value of 0.55).
# 
# Subjects encountered four different types of lists: 
# 1. Control lists that contained all once-presented items;  
# 2. pure massed lists containing all twice-presented items; 
# 3. pure spaced lists consisting of items presented twice at lags 1-8, where lag is defined as the number of intervening items between a repeated item's presentations; 
# 4. mixed lists consisting of once presented, massed and spaced items. Within each session, subjects encountered three lists of each of these four types. 
# 
# In each list there were 40 presentation positions, such that in the control lists each position was occupied by a unique list item, and in the pure massed and pure spaced lists, 20 unique words were presented twice to occupy the 40 positions. In the mixed lists 28 once-presented and six twice-presented words occupied the 40 positions. In the pure spaced lists, spacings of repeated items were chosen so that each of the lags 1-8 occurred with equal probability. In the mixed lists, massed repetitions (lag=0) and spaced repetitions (lags 1-8) were chosen such that each of the 9 lags of 0-8 were used exactly twice within each session. The order of presentation for the different list types was randomized within each session. For the first session, the first four lists were chosen so that each list type was presented exactly once. An experimenter sat in with the subject for these first four lists, though no subject had difficulty understanding the task.
# 
# The data for this experiment is stored in `data/repFR.mat`. Here we build structures from the dataset that works with our existing data analysis and fitting functions.

# %%
import scipy.io as sio
import numpy as np

# %%
path = 'data/raw/repFR.mat'
mat_file = sio.loadmat(path, squeeze_me=True)
mat_data = [mat_file['data'].item()[i] for i in range(14)]
list_length = mat_data[12]

pres_itemnos = mat_data[4].astype('int64')-1
presentations = []
for i in range(len(pres_itemnos)):
    seen = []
    presentations.append([])
    for p in pres_itemnos[i]:
        if p not in seen:
            seen.append(p)
        presentations[-1].append(seen.index(p))
presentations = np.array(presentations)

# discard intrusions, repeats from recalls
rec_itemnos = mat_data[2].astype('int64')-1
recalls = mat_data[6]
trials = []
trial_items = []
for i in range(len(recalls)):
    trials.append([])
    
    trial = list(recalls[i])
    for j in range(len(trial)):
        t = trial[j]
        trial_item = rec_itemnos[i][j]
        if (t > 0) and (t not in trials[-1]):
            trials[-1].append(t)
            trial_items.append(trial_item)
    
    while len(trials[-1]) < list_length:
        trials[-1].append(0)
        
trials = np.array(trials)

# other fields direct from mat file...

result = {
    'subject': np.expand_dims(mat_data[0].astype('int64'), axis=1),
    'session': np.expand_dims(mat_data[1].astype('int64'), axis=1),
    #'pres_items': mat_data[2],
    #'rec_items':  mat_data[2],
    'pres_itemnos': presentations,
    'pres_itemids': mat_data[4].astype('int64')-1,
    'rec_itemids': mat_data[2].astype('int64')-1,
    'recalls': trials,
    'listLength': np.expand_dims(np.ones(np.shape(mat_data[0]), dtype="int64"), axis=1) * 40,
    'list_type': np.expand_dims(mat_data[7].astype('int64'), axis=1),
    'irt': mat_data[3].astype('int64'),
    #'pres_lag': 
    #'recalls_lag':
    #'trial': mat_data[9].astype('int64'),
    #'intrusions': 
    #'subject_sess':
    #'massed_recalls':
}

result
