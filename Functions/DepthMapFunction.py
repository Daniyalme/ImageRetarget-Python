from MiDaS.run import run


def depth_map_function(image, model="dpt_large"):

    print("Calculating the Depth Map")
    depth = run(image, model)

    return depth
