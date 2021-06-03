import grpc
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import numpy as np

class TFServingClient(object):
    """This is a class for interacting with TFServing. 

    This class allows establishing a chennel with a TFServing endpoint
    and using it for efficient predictions. 

    Args:
        endpoint (str): TFServing's endpoint; should be in the form: '<hostname>:<port>';
                        if port isn't specified, default port will be used.
        model_name (str): Specific model being queried.
        default_port (int): gRPC endpoint runs on port 8500 by default.
        signature_name (str): optional.
        
    """
    def __init__(self, endpoint, model_name, default_port=8500, signature_name='serving_default'):
        self.model_name = model_name
        
        # Handle the case where port isn't specified
        if ~((":" in endpoint) and (endpoint.split(":")[-1].isnumeric()))
            endpoint = "%s:%d" % (endpoint, default_port)
 
        self.endpoint = endpoint
        self.channel = grpc.insecure_channel(self.endpoint)    
        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(self.channel)
        self.grpc_request = predict_pb2.PredictRequest()
        
        # The following needs to match <name> from "SINGULARITYENV_MODEL_NAME=<name>..."
        # used when singularity container running tfserving is launched  
        grpc_request.model_spec.name = self.model_name
       
        grpc_request.model_spec.signature_name = signature_name

    def predict(self, query_input, tensorize=True, timeout_s=10.0, layer_name="dense"):
        """ Query tfserving endpoint with specific input.

        Args:
             query_input (str): input that will be sent to tfserving. 
                                Assumes that input includes: 'atom', 'bond', 'connectivity'.
             tensorize (bool): flag for required tensorization/reshaping of components of input.
             timeout_s (int): request timeout in seconds.
             layer_name (str): name of the layer for which values will be returned.
        """ 

        if tensorize:
            query_input['atom'] = tf.make_tensor_proto(query_input['atom'], 
                                                       shape=[1, len(query_input['atom'])])
            query_input['bond'] = tf.make_tensor_proto(query_input['bond'], 
                                                       shape=[1, len(query_input['bond'])])
            query_input['connectivity'] = tf.make_tensor_proto(query_input['connectivity'], 
                                                               shape=[1] + list(np.array(query_input['connectivity']).shape))
        for c in ['atom', 'bond', 'connectivity']:
            self.grpc_request.inputs[c].CopyFrom(query_input[c]) 

        result = self.stub.Predict.future(self.grpc_request, timeout_s) 
        return result.outputs[layer_name]

