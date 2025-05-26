import numpy as np
import utils

label_methods = {
        'spinsvar': '\\mobius (Ours)', 
        'sparserc': 'SparseRC',
        'varlingam' : '\\varlingam',
        'd_varlingam' : '\\dlingam',
        'culingam' : '\\clingam',
        'dynotears': 'DYNOTEARS', 
        'nts-notears' : 'NTS-NOTEARS', 
        'tsfci' : 'tsFCI',
        'pcmci' : 'PCMCI',
        'TCDF': 'TCDF', 
        'lingam' : 'LiNGAM',
        'GES': 'GES', 
        'MMPC': 'MMHC', 
        'CAM' : 'CAM', 
        'FGS' : 'fGES',
        'sortnregress': 'sortnregress',
        'pc' : 'PC'
    }

if __name__ == '__main__':
    parser, args = utils.get_args()
    filename, _ = utils.get_filename(parser, args)
    methods = args.methods

    avg = {}
    std = {}
    for key in methods:
        avg[key] = []
        std[key] = []
    
    varsortability = []

    with open('results_UAI/AVG_{}.csv'.format(filename), 'r') as f:
        for line in f:
            info = line.split(',')
            for method in methods:
                if(info[0] == 'Acc {} is'.format(method)):
                    avg[method] = [float(info[1]), float(info[2]), float(info[3]), float(info[4]), float(info[5]), float(info[6]), float(info[7]), float(info[8]), float(info[9]), float(info[10]), float(info[11]), float(info[12]), float(info[13])]
                elif(info[0] == 'Std {} is'.format(method)):
                    std[method] = [float(info[1]), float(info[2]), float(info[3]), float(info[4]), float(info[5]), float(info[6]), float(info[7]), float(info[8]), float(info[9]), float(info[10]), float(info[11]), float(info[12]), float(info[13])]


    best_shd = min([avg[m][0] for m in methods])
    print(best_shd)
    best_tpr = max([avg[m][1] for m in methods])
    best_f1 = max([avg[m][5] for m in methods])
    best_auroc = max([avg[m][6] for m in methods])
    best_time = min([avg[m][8] for m in methods])

    for method in methods:
        result = '{} '.format(label_methods[method])
        # for ind in [0, 1, 5, 6, 8]:
        for ind in [0, 8]:
            if((avg[method][0] == best_shd and ind == 0) or 
            (avg[method][1] == best_tpr and ind == 1) or 
            (avg[method][5] == best_f1 and ind == 5) or
            (avg[method][6] == best_auroc and ind == 6) or
            (avg[method][8] == best_time and ind == 8)):
                result += '&  $\\bm' + '{' + '{:.2f}\pm{:.2f}'.format(avg[method][ind], std[method][ind]) + '}$  '
            else:
                result += '&  $    {:.2f}\pm{:.2f} $  '.format(avg[method][ind], std[method][ind]) 
        
        result += '\\\\ \n'
        with open('results_UAI/finance_table.tex', 'a') as f:
            f.write(result)
