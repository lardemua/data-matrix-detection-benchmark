training_params = {
    "priors_info":{
        "anchors" :[[[116, 90], [156,198], [373, 326]],
                    [[30, 61], [62, 45], [59, 119]],
                    [[10, 13], [16, 30], [33, 23]]],
        "classes": 10, 
    },
    "hyper_params": {
        "backbone_lr": 0.001,
        "base_lr": 0.01,
        "freeze_backbone": False,
        "decay_gamma": 0.1,
        "decay_step": 20,
    },
    "input_shape": {
        "height": 608,
        "width": 608,
    },
    "export_onnx": False,
    
}