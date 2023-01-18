import torch


class BaseModel(torch.nn.Module):
    def load(self, path):
        """Load model from file.

        Args:
            path (str): file path
        """
        parameters = torch.load(path, map_location=torch.device("cpu"))
        print("Loading weights")
        #if "optimizer" in parameters:
            #if "model" in parameters:
            #    parameters = parameters["model"]
           #else:
        # new_params = {}
        #parameters = parameters["state_dict"]
        # for k in parameters.keys():
        #     new_k = k[6:]
        #     new_params[new_k] = parameters[k]
        self.load_state_dict(parameters, strict=True)
