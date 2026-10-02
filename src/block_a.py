import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import pandas as pd
import operator
from functools import reduce



def BlockA(MF_key = "Triangular", T_norm_key = 'Product', S_norm_key = 'Algebric_Sum', defuzzying_key = 'centroid', H_V_k_crisp_input=0.5, H_L_k_crisp_input=0.5, P_bot_crisp_input = 0.333333333333333, P_top_crisp_input = 0.003):
    def min_t_norm(a,b):
        return min(a,b)

    def product_t_norm(a, b):
        return a * b

    def difference_t_norm(a, b):
        return max(0, a+b-1)

    def drastic_t_norm(a, b):
        return np.where(b == 1, a, np.where(a == 1, b, 0))
    
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
    l_trimf=[[0, 0, 0.5], [0, 0.5, 1], [0.5, 1, 1], [0.0, 0.0, 0.25], [0.0, 0.25, 0.5], [0.25, 0.5, 0.75], 
                    [0.5, 0.75, 1.0], [0.75, 1.0, 1.0], [0.0, 0.0, 0.14285714285714285], [0.0, 0.14285714285714285, 0.2857142857142857], 
                    [0.14285714285714285, 0.2857142857142857, 0.42857142857142855], [0.2857142857142857, 0.42857142857142855, 0.5714285714285714], 
                    [0.42857142857142855, 0.5714285714285714, 0.7142857142857142], [0.5714285714285714, 0.7142857142857142, 0.857142857142857], 
                    [0.7142857142857142, 0.8571428571428571, 1.0], [0.8571428571428572, 1.0, 1.0], [0.0, 0.3333333333333333, 0.6666666666666666]]

    l_guass = [0, 0.5, 1, 0.25, 0.75, 0.0, 0.14285714285714285,0.2857142857142857,0.42857142857142855, 
    0.5714285714285714, 0.7142857142857142, 
    0.8571428571428571 , 0.6666666666666666, 0.3333333333333333, 1.0]
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

    ## Inputs ##
    H_V_k =  ctrl.Antecedent(uni,'H_V_k')
    H_L_k =  ctrl.Antecedent(uni,'H_L_k')
    P_bot =  ctrl.Antecedent(uni,'P_bot')
    P_top =  ctrl.Antecedent(uni,'P_top')

    ## Ouputs ##
    Qvout_k =  ctrl.Consequent(uni,'Qvout_k')   
    Qlout_k = ctrl.Consequent(uni,'Qlout_k')


    ## Fuzzy sets ##
    if MF_key == "Triangular":
        for i in [ H_V_k, H_L_k, P_top]:
            i['L'] = fuzz.trimf(i.universe, [0, 0, 0.5])
            i['M'] = fuzz.trimf(i.universe, [0, 0.5, 1])
            i['H'] = fuzz.trimf(i.universe, [0.5, 1, 1])
    elif MF_key == "Gaussian":
        for i in [ H_V_k, H_L_k, P_top]:
            i['L'] = fuzz.gaussmf(i.universe,  0, 0.16666666666666666)
            i['M'] = fuzz.gaussmf(i.universe,  0.5, 0.16666666666666666)
            i['H'] = fuzz.gaussmf(i.universe,  1, 0.16666666666666666)
    if MF_key == "Triangular":  
        for i in [ P_bot]:
            i['L'] = fuzz.trimf(i.universe, [0.0, 0.0, 0.3333333333333333])
            i['M'] = fuzz.trimf(i.universe, [0.0, 0.3333333333333333, 0.6666666666666666])
            i['H'] = fuzz.trimf(i.universe, [0.3333333333333333, 0.6666666666666666, 1.0])
            i['VH'] = fuzz.trimf(i.universe, [0.6666666666666667, 1.0, 1.0])
    if MF_key == "Gaussian":
        for i in [ P_bot]:    
            i['L'] = fuzz.gaussmf(i.universe, 0 ,0.1111111111111111)
            i['M'] = fuzz.gaussmf(i.universe,  0.3333333333333333, 0.1111111111111111)
            i['H'] = fuzz.gaussmf(i.universe,  0.6666666666666666, 0.1111111111111111)
            i['VH'] = fuzz.gaussmf(i.universe, 1, 0.1111111111111111)

    if MF_key == "Triangular":
        Qvout_k_trimf = {'L': [0.0, 0.0, 0.25], 'M-L': [0.0, 0.25, 0.5], 'M': [0.25, 0.5, 0.75], 
                        'H-M': [0.5, 0.75, 1.0], 'H': [0.75, 1.0, 1.0]}
    if MF_key == "Gaussian":
        Qvout_k_gauss = {'L': [0.0, 0.08333333333333333], 'M-L': [0.25, 0.08333333333333333], 
    'M': [0.5, 0.25], 'H-M': [0.75, 0.08333333333333333], 'H': [1.0, 0.08333333333333333]}
    if MF_key == "Triangular":
        Qlout_k_trimf = {'VL': [0.0, 0.0, 0.14285714285714285], 'L-VL': [0.0, 0.14285714285714285, 0.2857142857142857], 
                        'L': [0.14285714285714285, 0.2857142857142857, 0.42857142857142855], 'M-L': [0.2857142857142857, 0.42857142857142855, 0.5714285714285714], 
                        'M': [0.42857142857142855, 0.5714285714285714, 0.7142857142857142], 'H-M': [0.5714285714285714, 0.7142857142857142, 0.857142857142857], 
                        'H': [0.7142857142857142, 0.8571428571428571, 1.0], 'VH-H': [0.8571428571428572, 1.0, 1.0]}
    if MF_key == "Gaussian":
        Qlout_k_gauss = {'VL': [0.0, 0.047619047619047616], 'L-VL': [0.14285714285714285, 0.047619047619047616], 
    'L': [0.2857142857142857, 0.047619047619047616], 'M-L': [0.42857142857142855, 0.047619047619047616], 
    'M': [0.5714285714285714, 0.047619047619047616], 'H-M': [0.7142857142857142, 0.047619047619047616], 
    'H': [0.8571428571428571, 0.047619047619047616], 'VH-H': [1.0, 0.047619047619047616]}



    if MF_key == "Gaussian":
        for k,v in Qvout_k_gauss.items():
            Qvout_k[k] = fuzz.gaussmf(Qvout_k.universe, v[0], v[1])
        for k,v in Qlout_k_gauss.items():
            Qlout_k[k] = fuzz.gaussmf(Qlout_k.universe, v[0], v[1])
    elif MF_key == "Triangular":
        for k,v in Qvout_k_trimf.items():
            Qvout_k[k] = fuzz.trimf(Qvout_k.universe, v)
        for k,v in Qlout_k_trimf.items():
            Qlout_k[k] = fuzz.trimf(Qlout_k.universe, v)

    #Defuzzifying method
    #defuzzifying_method_list = ['centroid', 'bisector', 'mom', 'som', 'lom'] 
    Qvout_k.defuzzify_method = defuzzying_key
    Qlout_k.defuzzify_method = defuzzying_key


    df = pd.read_csv("Rules_of_BlockA.csv")


    variable_dic = {'H_V_k':H_V_k, 'H_L_k':H_L_k, 
                    'P_bot':P_bot, 'P_top':P_top, 'Qvout_k':Qvout_k, 
                    'Qlout_k':Qlout_k}

   
    rules = []
    for _,row in df.iterrows():
        base = None
        for col, term in row.items():
            if col in ("ID") or pd.isna(term):
                continue
            if col in ("Qvout_k", "Qlout_k"):
                continue
            expr = variable_dic[col][str(term).strip()]
            base = expr if base is None else (base & expr)
        
        antecendent = base
        consequent = []
        for col in ("Qvout_k", "Qlout_k"):
            term = row.get(col)
            if not pd.isna(term):
                consequent.append(variable_dic[col][str(term).strip()])
        
        if consequent:
            rule = ctrl.Rule(antecendent, consequent, label = str(row['ID']), and_func=T_norm_fun, or_func=S_norm_fun)

            rules.append(rule)



    ctrl_sys = ctrl.ControlSystem(rules)
    sim = ctrl.ControlSystemSimulation(ctrl_sys)

    Input_names = [ 'H_V_k', 'H_L_k', 'P_bot', 'P_top']



    Inputs = [ H_V_k_crisp_input, H_L_k_crisp_input, P_bot_crisp_input, P_top_crisp_input]

    Inputs_dic={}
    for i in range(len(Inputs)):
        Inputs_dic[Input_names[i]] = Inputs[i]



 
    for k,v in Inputs_dic.items():
        sim.input[k] = float(v)
    try:
        sim.compute()
        Qvout_k_out = sim.output['Qvout_k']
        Qlout_k_out = sim.output['Qlout_k']
    except Exception as e:
        print("The model has baldness")
        
    return Qvout_k_out, Qlout_k_out


