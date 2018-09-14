"""The model swap function as well as the function for initial guesses."""

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
        if "all" in name:
            a1,a2,b1,b2,c1,c2 = params
        else:
            raise Exception("Single snapshot model swap not implemented.")
    else:
        raise Exception("Model swap not implemented.")
    return a1, a2, b1, b2, c1, c2

def initial_guess(args):
    name = args['name'] #name of the model
    if name == "sflinear_combinations":
        #Try the kept values of the model defaults
        #Premade at the args step
        return args['defaults'][args['kept']]
    if "single_snapshot" in name:
        if "all" in name:
            #Try the Tinker defaults
            y = np.log10(200)
            a1,a2 = 1+.24*y*np.exp(-(4/y)**4), 0.44*y-0.88
            b1,b2 = 0.183, 1.5
            c1 = 0.019+0.107*y+0.19*np.exp(-(4/y)**4)
            c2 = 2.4
            return np.array([a1, a2, b1, b2, c1, c2])
