training_params = {
    "priors_info":{
        "anchors" :[[[1.5972652, 2.063394], [2.7095582,3.3936112], [4.9832897, 6.391053]],
                     [[0.66488904, 1.1158209], [1.0766983, 1.4735554], [1.15916838, 0.6244885]],
                     [[0.179995492, 0.329541], [0.44646746, 0.7362717], [0.47200382, 0.37327695]]],
        "classes": 2, 
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