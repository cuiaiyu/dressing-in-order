import collections

ATR_LABEL = ['Background', 'Hat', 'Hair', 'Sunglasses', 
             'Upper-clothes', 'Skirt', 'Pants', 'Dress', 
             'Belt', 'Left-shoe', 'Right-shoe', 'Face', 
             'Left-leg', 'Right-leg', 'Left-arm', 'Right-arm', 'Bag', 'Scarf']

YF_LABEL = ['Background','Pants','Hair','Skirt','Face','Upper-clothes','Arm','Leg']
DF_LABEL = YF_LABEL #
AI_LABEL = ['Background','Face','Arm','Leg','Hair','Upper-clothes','Pants','Skirt',]
ATR_TO_DF = {
    'Upper-clothes': 'Upper-clothes', 
    'Skirt': 'Skirt', 
    'Pants': 'Pants', 
    'Dress': 'Upper-clothes',
    'Face':'Face',
    'Left-leg':'Arm',
    'Right-leg':'Arm',
    'Left-arm':'Leg',
    'Right-arm':'Leg',
    'Hair':'Hair'
   
}
LIP_LABEL = ['Background', 'Hat','Hair','Glove','Sunglasses',
'Upper-clothes','Dress','Coat','Socks','Pants',
'Jumpsuits','Scarf','Skirt','Face','Left-arm',
'Right-arm','Left-leg','Right-leg','Left-shoe','Right-shoe']

LIP_TO_DF = {
    'Upper-clothes': 'Upper-clothes', 
    'Coat':'Skirt',
    'Skirt': 'Pants', 
    'Pants': 'Pants', 
    'Dress': 'Upper-clothes',
    'Face':'Face',
    'Left-leg':'Arm',
    'Right-leg':'Arm',
    'Left-arm':'Leg',
    'Right-arm':'Leg',
    'Hair':'Hair'   
}

def get_df_to_aiyu():
    df2aiyu = {DFS.index(i):AIYU_LABEL_6.index(DF_TO_AIYU[i]) for i in DFS}
    return df2aiyu
   
def get_label_map(n_human_part=8, label_set='lip'):
    if label_set == 'lip':
        atr2aiyu = {LIP_LABEL.index(i): DF_LABEL.index(LIP_TO_DF[i]) if i in LIP_TO_DF else 0 for i in LIP_LABEL}
    else:
        atr2aiyu = {ATR_LABEL.index(i): DF_LABEL.index(ATR_TO_DF[i]) if i in ATR_TO_DF else 0 for i in ATR_LABEL}
    aiyu2atr = collections.defaultdict(list)
    for atr in atr2aiyu:
        aiyu = atr2aiyu[atr]
        aiyu2atr[aiyu].append(atr)
    return aiyu2atr, atr2aiyu