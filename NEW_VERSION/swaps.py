"""The model swap function as well as the function for initial guesses."""
import numpy as np

#Tinker defaults
y = np.log10(200)
a1,a2 = 1+.24*y*np.exp(-(4/y)**4), 0.44*y-0.88
b1,b2 = 0.183, 1.5
c1 = 0.019+0.107*y+0.19*np.exp(-(4/y)**4)
c2 = 2.4
tds = np.array([a1,a2,b1,b2,c1,c2])

def model_swap(params, args, x):
    """
    Used to swap between different halo bias models.
    x might contain some kind of dependence on epoch.
    """
    name = args['name'] #name of the model

    if name == "sflinear_combinations":
        """Linear in scale factor
        Defined by combinations of the 12 possible parameters
        """
        #Set the parameters we keep by the 'kept' array
        pars[args['kept']] = params
        #Set the parameters we freeze with the 'dropped' array,
        #setting them to their default values
        pars[args['dropped']] = args['defaults'][args['dropped']]
        a1,a2,b1,b2,c1,c2 = pars[:6] + x*pars[6:]
    elif "single_snapshot" in name:
        out = None
        if "all" in name:
            out = params
        elif "m1" in name:
            a1 = tds[0]
            a2,b1,b2,c1,c2 = params
            out = np.array([a1,a2,b1,b2,c1,c2])
        elif "m2" in name:
            a1 = tds[0]
            b1 = tds[2]
            a2,b2,c1,c2 = params
            out = np.array([a1,a2,b1,b2,c1,c2])
        elif "m3" in name:
            a1,a2,b1 = tds[:3]
            b2,c1,c2 = params
            out = np.array([a1,a2,b1,b2,c1,c2])
        elif "m4" in name:
            a1,a2,b1,b2 = tds[:4]
            c1,c2 = params
            out = np.array([a1,a2,b1,b2,c1,c2])
        elif "m5" in name:
            a1 = tds[0]
            b2 = tds[3]
            a2,b1,c1,c2 = params
            out = np.array([a1,a2,b1,b2,c1,c2])
        elif "m6" in name:
            a1 = tds[0]
            c1 = tds[4]
            a2,b1,b2,c2 = params
            out = np.array([a1,a2,b1,b2,c1,c2])

        if out is None:
            raise Exception("Single snapshot model swap not implemented.")
        else:
            return out
    else:
        raise Exception("Model swap not implemented.")
    return a1, a2, b1, b2, c1, c2

def initial_guess(args):
    name = args['name'] #name of the model
    if name == "sflinear_combinations":
        #Try the kept values of the model defaults
        #Premade elsewhere
        return args['defaults'][args['kept']]
    elif "single_snapshot" in name:
        out = None
        if "all" in name:
            out = tds
        elif "m1" in name:
            out = tds[1:]
        elif "m2" in name:
            out = np.delete(tds, [0,2])
        elif "m3" in name:
            out = np.delete(tds, [0,1,2])
        elif "m4" in name:
            out = np.delete(tds, [0,1,2,3])
        elif "m5" in name:
            out = np.delete(tds, [0,3])
        elif "m6" in name:
            out = np.delete(tds, [0,4])
        if out is None:
            raise Exception("Single snap model initial guess not implemented.")
        else:
            return out
            
    else:
        raise Exception("Model initial guess not implemented.")
