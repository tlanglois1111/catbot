import uff

uff_model = uff.from_tensorflow_frozen_model(frozen_file='frozen_inference_graph.pb',
                                             out_filename='catbot.engine',
                                             text=True,
                                             list_nodes=False,
                                             debug_mode=False)

