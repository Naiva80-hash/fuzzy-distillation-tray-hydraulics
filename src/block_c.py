import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import pandas as pd


def BlockC(MF_key = "Triangular", T_norm_key = 'Product', S_norm_key = 'Algebric_Sum', defuzzying_key = 'centroid', Qvin_k_crisp_input = 0.6 , Qvout_k_crisp_input = 0.1 , HV_k_crisp_input = 0.5):
    
    def min_t_norm(a,b):
        return min(a,b)

    def product_t_norm(a, b):
        return a * b

    def difference_t_norm(a, b):
        return max(0, a+b-1)

    def drastic_t_norm(a, b):
        return np.where(b == 1, a, np.where(a == 1, b, 0))
    #custom s-norms
    def max_s_norm(a,b):
        return max(a,b)

    def Bounded_sum_s_norm(a, b):
        return min(1, a+b)

    def Algebric_sum_s_norm(a, b):
        return a+b-a*b

    def drastic_s_norm(a, b):
        return np.where(a == 0, b, np.where(b == 0, a, 1))


    if T_norm_key == 'Product':
        T_norm_fun = product_t_norm
    elif T_norm_key == "Difference":
        T_norm_fun = difference_t_norm
    elif T_norm_key == "Drastic":
        T_norm_fun = drastic_t_norm
    else:
        T_norm_fun = min_t_norm

    if S_norm_key == 'Drastic':
        S_norm_fun = drastic_s_norm
    elif S_norm_key == "Bounded_Sum":
        S_norm_fun = Bounded_sum_s_norm
    elif S_norm_key == "Algebric_Sum":
        S_norm_fun = Algebric_sum_s_norm
    else:
        S_norm_fun = max_s_norm



    base = np.linspace(0,1,100)
    extra_trimf_list = []
    extra_guass_list = []
    l_trimf=[[0.0, 0.0, 0.125],[0.0, 0.125, 0.25], [0.125, 0.25, 0.375], 
                    [0.25, 0.375, 0.5], [0.375, 0.5, 0.625], 
                    [0.5, 0.625, 0.75], [0.625, 0.75, 0.875], 
                    [0.75, 0.875, 1.0], [0.875, 1.0, 1.0]]
    l_guass = [0.0, 0.125, 
                    0.25, 0.375, 
                    0.5, 0.625, 0.75, 
                    0.875, 1.0]

    for l in l_trimf:
        for v in l:
            if v not in extra_trimf_list:
                extra_trimf_list.append(v)
    for v in l_guass:
        if v not in extra_guass_list:
            extra_guass_list.append(v)
    extra_guass = np.array(extra_guass_list)
    extra_trimf = np.array(extra_trimf_list)

    if MF_key == "Triangular":
        uni = np.sort(np.unique(np.concatenate((base,extra_trimf))))
    if MF_key == "Gaussian":
        uni = np.sort(np.unique(np.concatenate((base,extra_guass))))

    ## ## Universe deltaQ####
    base = np.linspace(-1,1,100)
    extra_trimf_list_deltaQ = []
    extra_guass_list_deltaQ = []
    l_trimf_deltaQ=[[-1.0, -1.0, -0.5],
            [-1.0, -0.5, 0.0],[-0.5, 0.0, 0.5],[0.0, 0.5, 1.0],
            [0.5, 1.0, 1.0]]
    l_guass_deltaQ = [-1, -0.5, 0, 1, 0.5]

    for l in l_trimf_deltaQ:
        for v in l:
            if v not in extra_trimf_list_deltaQ:
                extra_trimf_list_deltaQ.append(v)
    for v in l_guass_deltaQ:
        if v not in extra_guass_list_deltaQ:
            extra_guass_list_deltaQ.append(v)
    extra_guass_deltaQ = np.array(extra_guass_list_deltaQ)
    extra_trimf_deltaQ = np.array(extra_trimf_list_deltaQ)

    if MF_key == "Triangular":
        uni_deltaQ = np.sort(np.unique(np.concatenate((base,extra_trimf_deltaQ))))
    if MF_key == "Gaussian":
        uni_deltaQ = np.sort(np.unique(np.concatenate((base,extra_guass_deltaQ))))


    deltaQV = ctrl.Antecedent(uni_deltaQ, 'deltaQV')
    HV_k=  ctrl.Antecedent(uni,'HV_k')



    ## Ouputs ##
    HV_k1 =  ctrl.Consequent(uni,'HV_k1')   



    ## Fuzzy sets ##
    if MF_key == "Triangular":
        for i in [deltaQV]:
            i['NL'] = fuzz.trimf(i.universe, [-1.0, -1.0, -0.5])
            i['N'] = fuzz.trimf(i.universe, [-1.0, -0.5, 0.0])
            i['Z'] = fuzz.trimf(i.universe, [-0.5, 0.0, 0.5])
            i['P'] = fuzz.trimf(i.universe, [0.0, 0.5, 1.0])
            i['PL'] = fuzz.trimf(i.universe, [0.5, 1.0, 1.0])
    elif MF_key == "Gaussian":
        for i in [deltaQV]:
            i['NL'] = fuzz.gaussmf(i.universe,  -1, 0.21233045007200477)
            i['N'] = fuzz.gaussmf(i.universe,  -0.5, 0.21233045007200477)
            i['Z'] = fuzz.gaussmf(i.universe,  0, 0.21233045007200477)
            i['P'] = fuzz.gaussmf(i.universe,  0.5, 0.21233045007200477)
            i['PL'] = fuzz.gaussmf(i.universe,  1, 0.21233045007200477)


    if MF_key == "Triangular":
        HV_k_trimf = {'VL': [0.0, 0.0, 0.125], 'L-VL': [0.0, 0.125, 0.25], 'L': [0.125, 0.25, 0.375], 
                    'M-L': [0.25, 0.375, 0.5], 'M': [0.375, 0.5, 0.625], 
                    'H-M': [0.5, 0.625, 0.75], 'H': [0.625, 0.75, 0.875], 
                    'VH-H': [0.75, 0.875, 1.0], 'VH': [0.875, 1.0, 1.0]}
        
    if MF_key == "Gaussian":
        HV_k_gauss = {'VL': [0.0, 0.05308261251800119], 'L-VL': [0.125, 0.05308261251800119], 
                    'L': [0.25, 0.05308261251800119], 'M-L': [0.375, 0.05308261251800119], 
                    'M': [0.5, 0.05308261251800119], 'H-M': [0.625, 0.05308261251800119], 'H': [0.75, 0.05308261251800119], 
                    'VH-H': [0.875, 0.05308261251800119], 'VH': [1.0, 0.05308261251800119]}



    if MF_key == "Gaussian":
        for k,v in HV_k_gauss.items():
            HV_k[k] = fuzz.gaussmf(HV_k.universe, v[0], v[1])
        for k,v in HV_k_gauss.items():
            HV_k[k] = fuzz.gaussmf(HV_k.universe, v[0], v[1])
    elif MF_key == "Triangular":
        for k,v in HV_k_trimf.items():
            HV_k[k] = fuzz.trimf(HV_k.universe, v)
        for k,v in HV_k_trimf.items():
            HV_k[k] = fuzz.trimf(HV_k.universe, v)


    if MF_key == "Triangular":
        HV_k1_trimf = {'VL': [0.0, 0.0, 0.125], 'L-VL': [0.0, 0.125, 0.25], 'L': [0.125, 0.25, 0.375], 
                    'M-L': [0.25, 0.375, 0.5], 'M': [0.375, 0.5, 0.625], 
                    'H-M': [0.5, 0.625, 0.75], 'H': [0.625, 0.75, 0.875], 
                    'VH-H': [0.75, 0.875, 1.0], 'VH': [0.875, 1.0, 1.0]}
    if MF_key == "Gaussian":
        HV_k1_gauss = {'VL': [0.0, 0.05308261251800119], 'L-VL': [0.125, 0.05308261251800119], 
                    'L': [0.25, 0.05308261251800119], 'M-L': [0.375, 0.05308261251800119], 
                    'M': [0.5, 0.05308261251800119], 'H-M': [0.625, 0.05308261251800119], 'H': [0.75, 0.05308261251800119], 
                    'VH-H': [0.875, 0.05308261251800119], 'VH': [1.0, 0.05308261251800119]}



    if MF_key == "Gaussian":
        for k,v in HV_k1_gauss.items():
            HV_k1[k] = fuzz.gaussmf(HV_k1.universe, v[0], v[1])
        for k,v in HV_k1_gauss.items():
            HV_k1[k] = fuzz.gaussmf(HV_k1.universe, v[0], v[1])
    elif MF_key == "Triangular":
        for k,v in HV_k1_trimf.items():
            HV_k1[k] = fuzz.trimf(HV_k1.universe, v)
        for k,v in HV_k1_trimf.items():
            HV_k1[k] = fuzz.trimf(HV_k1.universe, v)


    #Defuzzifying method
    HV_k1.defuzzify_method = defuzzying_key
    HV_k1.defuzzify_method = defuzzying_key


    ## Reading rules ##
    df = pd.read_csv("Rules_deltaQ_BlockC.csv")


    variable_dic = {'deltaQV':deltaQV, 'HV_k':HV_k, "HV_k1":HV_k1}

    eqiv_P = {}
    rules = []
    for _,row in df.iterrows():
        base = None
        for col, term in row.items():
            if col in ("ID") or pd.isna(term):
                continue
            if col =="HV_k1":
                continue
            expr = variable_dic[col][str(term).strip()]
            base = expr if base is None else (base & expr)
        
    
        antecendent = base
        consequent = []
        if col == "HV_k1":
            term = row.get(col)
            if not pd.isna(term):
                consequent.append(variable_dic[col][str(term).strip()])
        
        if consequent:
            rule = ctrl.Rule(antecendent, consequent, label = str(row['ID']), and_func=T_norm_fun, or_func=S_norm_fun)

            rules.append(rule)



    ctrl_sys = ctrl.ControlSystem(rules)
    sim = ctrl.ControlSystemSimulation(ctrl_sys)




    deltaQV_crisp_input = float(Qvin_k_crisp_input) - float(Qvout_k_crisp_input)



    sim.input['HV_k'] = float(HV_k_crisp_input)
    sim.input["deltaQV"] = float(deltaQV_crisp_input)
    try:
        sim.compute()
        HV_k1_out = sim.output['HV_k1']
    except Exception as e:
        print("The model has baldness")
            
    return HV_k1_out

