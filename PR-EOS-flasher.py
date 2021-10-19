import math
import decimal
import simplejson as json
import scipy
import ast
import boto3
from scipy.optimize import fsolve
from functools import reduce
import functools
from boto3.dynamodb.conditions import Key, Attr



lmap = lambda func, *iterable: list(map(func, *iterable))
flatten_dict = lambda x: {k: v for d in x for k, v in d.items()}

def compose(*functions):
    return functools.reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)


def process_DB_response(res):
    eval_fn = lambda x: ast.literal_eval((json.dumps(x, use_decimal=True)))
    return lmap(eval_fn, res["Items"])

def DB_flat_to_lookup_dict(result_flat):
    lookup_key = result_flat["ComponentName"]
    return {lookup_key:result_flat}

# read component properties and binary interaction parameters from database
dynamodb = boto3.resource('dynamodb', region_name="us-east-1")
table = dynamodb.Table('MCFlashComponentProperties')
response = table.scan()
rezzy = process_DB_response(response)
lookup_dict = lmap(DB_flat_to_lookup_dict, rezzy)
flatten_lookup = flatten_dict(lookup_dict)

# test data
test_input ={'Temperature':200,
             'Pressure':1800,
             'run_id': 1,
             'Mixture_Weight' : {'H2S': 1.368,
                                 'CH4': 10.238,
                                 'i-C4':20.616,
                                 'nC8': 27.91,
                                 'nC15':39.862
                                },
             'ComponentProperties' : flatten_lookup
            }


def add_wilson_correlation(componentName, in_data = test_input):
    componentProps = in_data["ComponentProperties"][componentName]
    temp_rankine = in_data["Temperature"] + 460
    press = in_data["Pressure"]
    T_ci = componentProps["Tc"] + 460
    P_ci = componentProps["Pc"]
    T_ri = temp_rankine / T_ci
    P_ri = press / P_ci
    w = componentProps["wAcentric"]
    K_wilson = math.exp(5.37 * (1+w) * ( 1 - 1/T_ri)) / P_ri
    componentProps["K_i"] = K_wilson
    return  {componentName: componentProps}

def calc_feed_mols(componentName, in_data = test_input):
    componentProps = in_data["ComponentProperties"][componentName]
    mw = componentProps["MW"]
    mix_wt_frac = in_data['Mixture_Weight'][componentName] / 100
    mols = (mw / mix_wt_frac)**(-1)
    componentProps['mols'] = mols
    return {componentName: componentProps}

def calc_total_mols(in_data):
    vals = in_data.values()
    n_mols_tot = sum( lmap( lambda x: x["mols"] , vals))
    return n_mols_tot

def add_mol_frac(componentName, in_data):
    #in_data is a dict of dicts of form {"compName":compProps}
    these_vals = in_data[componentName]
    total_mols = calc_total_mols(in_data)
    mol_frac = these_vals["mols"] / total_mols
    these_vals["z_i"] = mol_frac
    return {componentName : these_vals}

component_ls = test_input['Mixture_Weight'].keys()
aug1 = lmap(add_wilson_correlation, component_ls)
aug2 = flatten_dict(lmap(calc_feed_mols, component_ls))
aug3 = flatten_dict(lmap(lambda x: add_mol_frac(x,aug2), component_ls))

test_input["ComponentProperties"] = aug3

###-----------------------------------------------------------######

def rashford_rice(full_input):
    # input will be full test_input style dict
    compProps = full_input["ComponentProperties"]
    def func_rr(x):
        return  sum( lmap( lambda y: y["z_i"] * (1 - y["K_i"]) / (y["K_i"] + x[0] * ((1 - y["K_i"]) )), compProps.values()))
    L_root = fsolve(func_rr, [0.5])[0]
    print(L_root)
    V = 1 - L_root
    xi = lmap( lambda x: x["z_i"] / (x["K_i"] + L_root * (1 -x["K_i"])), compProps.values())
    yi = lmap( lambda x: x["K_i"] * x["z_i"] / (x["K_i"] + L_root * (1 -x["K_i"])), compProps.values())
    consolidated_x_y = zip(compProps.keys(), xi,yi)
    
    def update_component_props(nxy_tuple):
        (cName, xi, yi) = nxy_tuple
        compProps[cName]["x_i"] = xi
        compProps[cName]["y_i"] = yi
        return ## hmm not a pure function i guess whatever let's see if it makes debugging harder later

    update_in_place = lmap(update_component_props, consolidated_x_y)
    full_input["ComponentProperties"] = compProps
    return full_input
    

R_ig = 10.731 # ft^3 psi lb-mol R^-1

def alpha_fn(w,T, Tc, Pc):
    if w < 0.5:
        m = 0.3764 + 1.54226*w -0.26992*(w**2)
    else:
        m = 0.37960 +1.485*w - 0.1644*(w**2) + 0.01667*(w**3)
    T_r = T / Tc
    alf = (1 + m * (1-math.sqrt(T_r)))**2
    return alf

def a_fn(Tc,Pc):
    return 0.457235 *  R_ig**2 * Tc**2 / Pc

def b_fn(Tc,Pc):
    return 0.077796  * R_ig * Tc / Pc

def add_PR_pc_properties(full_input):
    temp = full_input["Temperature"] + 460
    press = full_input["Pressure"]
    compProps = full_input["ComponentProperties"]
    def add_pc_props(sing_component):
        w_acen = sing_component["wAcentric"]
        Tc = sing_component["Tc"] + 460
        Pc = sing_component["Pc"]
        sing_component["alpha_i"] = alpha_fn(w_acen, temp, Tc, Pc)
        sing_component["a_i"] = a_fn(Tc, Pc)
        sing_component["b_i"] = b_fn(Tc,Pc)
        sing_component["B_i"] = sing_component["b_i"] * press / (R_ig * temp)
        return {sing_component["ComponentName"]:sing_component}
    props_update = lmap(add_pc_props, compProps.values())
    full_input["ComponentProperties"] = flatten_dict( props_update )
    return full_input


def PR_mixture(full_input, vl_toggle = "Vapor"):
    compProps = full_input["ComponentProperties"]
    press = full_input["Pressure"]
    temp = full_input["Temperature"] + 460

    def b_mixture(vap_liq="Vapor"):
        if vap_liq == "liquid":
            return sum(lmap(lambda x: x["x_i"]* x["b_i"] , compProps.values()))
        else:
            return sum(lmap(lambda x: x["y_i"]* x["b_i"] , compProps.values()))
    
    def a_alf_mixture(vap_liq="Vapor"):
        # compProps is in scope
        def inner_fn(cName, outer):
            nj = compProps[cName]
            kij = outer["BIP-PR"][cName]
            if vap_liq == "liquid":
                y_j = nj["x_i"]
                y_i = outer["x_i"]
            else:
                y_j = nj["y_i"]
                y_i = outer["y_i"]
            return y_i * y_j * (1-kij) * math.sqrt(outer["a_i"] * outer["alpha_i"] * nj["a_i"] * nj["alpha_i"] )
        return sum(lmap(lambda ni: sum(lmap( lambda j_name: inner_fn(j_name,ni)  , ni["BIP-PR"].keys()))
                    , compProps.values()))

    mix_b = b_mixture(vl_toggle)
    mix_a_alf = a_alf_mixture(vl_toggle)
    A_mix = mix_a_alf * press / (R_ig**2 * temp**2)
    B_mix = mix_b * press / (R_ig * temp)

    def Z_root_fn(A,B):
        def Z_poly(x):
            return x[0]**3 - (1-B)* x[0]**2 + (A - 2*B -3*B**2)*x[0] - (A*B - B**2 - B**3)
        z_root = fsolve(Z_poly,[0.8])[0]
        return z_root

    z_factor = Z_root_fn(A_mix, B_mix)
    V_mix = z_factor * R_ig * temp / press
    return {"Z-factor": z_factor,
            "V_mix": V_mix,
             "A": A_mix,
             "B": B_mix,
             "a_alpha": mix_a_alf}

pipe = compose(add_PR_pc_properties, rashford_rice)
qq = pipe(test_input)

def calc_mix_props(full_input, vap_liq):
    return {"Phase": vap_liq,
            "PR-Results" : PR_mixture(full_input,vap_liq)}

mixture_props = lmap(lambda x:calc_mix_props(qq,x),["vapor","liquid"])

def add_partial_fugacity( mixture_res, full_component_data):
    compProps = full_component_data["ComponentProperties"]
    compKeys = compProps.keys()
    press = full_component_data["Pressure"]

    def calc_sc_partial_fugacity(cName, phase):
        B_i = compProps[cName]["B_i"]
        B = filter(lambda x:x["Phase"] == phase, mixture_res)[0]["PR-Results"]["B"]
        A = filter(lambda x:x["Phase"] == phase, mixture_res)[0]["PR-Results"]["A"]
        a_alfa = filter(lambda x:x["Phase"] == phase, mixture_res)[0]["PR-Results"]["a_alpha"]
        Z = filter(lambda x:x["Phase"] == phase, mixture_res)[0]["PR-Results"]["Z-factor"]
        if phase == "liquid":
            part = compProps[cName]["x_i"]
        else:
            part = compProps[cName]["y_i"]
        
        def calc_s_term(cName, phase):
            a_alf_i = compProps[cName]["a_i"] * compProps[cName]["alpha_i"]
            kijs = compProps[cName]["BIP-PR"]
            if phase == "liquid":
                phase_key = "x_i"
            else:
                phase_key = "y_i"
            return sum(lmap( lambda k: compProps[k][phase_key] * (1-kijs[k]) * math.sqrt(a_alf_i * compProps[k]["a_i"] * compProps[k]["alpha_i"])
                  ,kijs.keys()))
 
        s_term = calc_s_term(cName, phase)
        term1 = (B_i / B) * (Z-1) 
        term2 = math.log(Z-B)
        term3f = A / (2 * math.sqrt(2) * B)
        term3b = (B_i / B) - 2 / (a_alfa) * s_term
        term3r = math.log((Z + B *(1 + math.sqrt(2))) / (Z + B *(1 - math.sqrt(2))))

        ins = term1 - term2 + term3f * term3b * term3r
        part_fugacity = part * press * math.exp(ins)  

        return {cName: part_fugacity}

    liq_fugacities = flatten_dict(lmap(lambda x: calc_sc_partial_fugacity(x,"liquid"), compKeys))
    vap_fugacities = flatten_dict(lmap(lambda x: calc_sc_partial_fugacity(x,"vapor"), compKeys))

    liq_mix = filter(lambda x:x["Phase"] == 'liquid', mixture_res)[0]
    liq_mix["Partial Fugacities"] = liq_fugacities
    vap_mix = filter(lambda x:x["Phase"] == 'vapor', mixture_res)[0]
    vap_mix["Partial Fugacities"] = vap_fugacities
    return [vap_mix, liq_mix]

def fugacity_convergence(mixture_res, full_input):
    def check_fugacity_eq(mixture_res):
        vap_fugacities = filter(lambda x:x["Phase"]=="vapor", mixture_res)[0]["Partial Fugacities"]
        liq_fugacities = filter(lambda x:x["Phase"]=="liquid", mixture_res)[0]["Partial Fugacities"]
        fug_ratios = lmap(lambda x: liq_fugacities[x] / vap_fugacities[x] , vap_fugacities.keys())
        fug_check = lmap(lambda x: vap_fugacities[x] / liq_fugacities[x] , vap_fugacities.keys())
        zero_check = len(filter(lambda z: z>10e-8 , lmap(lambda x: x-1, fug_check)))
        fug_dict = dict(zip(vap_fugacities.keys(), fug_ratios))
        return (fug_dict, zero_check)

    def update_k_values(new_k_multipliers, mix_set):
        cProps = mix_set["ComponentProperties"]
        def update_1_k(cName):
            this_cProp = cProps[cName]
            this_new_k_multiplier =  new_k_multipliers[cName]
            this_cProp["K_i"] = this_cProp["K_i"]  * this_new_k_multiplier
            cProps[cName] = this_cProp
            return cProps
        new_cProps = [ update_1_k(k) for k in new_k_multipliers.keys() ][-1] # don't really like this updating approach, it'll prolly work but ehh
        mix_set["ComponentProperties"] = new_cProps
        return mix_set


    n_mix_set = full_input
    check_t = check_fugacity_eq(mixture_res)
    n_k_multipliers = check_t[0]
    conv_check = check_t[1]
    
    while conv_check > 0:
        n_mix_set = update_k_values(n_k_multipliers, n_mix_set)
        n_mix_set = rashford_rice(n_mix_set)
        n_mix_props = add_partial_fugacity(lmap(lambda x:calc_mix_props(n_mix_set,x),["vapor","liquid"]), n_mix_set)
        check_t = check_fugacity_eq(n_mix_props)
        n_k_multipliers = check_t[0]
        conv_check = check_t[1]

    return (n_mix_props,n_mix_set)

mix_setup =  add_partial_fugacity(mixture_props, qq)
converge_mix = fugacity_convergence(mix_setup, qq)