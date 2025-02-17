import os
import onnx
from onnx import helper, TensorProto
import onnxruntime

WAY_TO_YOLOV8 = "../base_model/yolov8/"
MODEL_TYPE = "yolov8n_original.onnx"

class Informator:
    @classmethod
    def get_info(cls,model: onnx.ModelProto):
        print("ALL nodes of YOLOV8:")
        for node in model.graph.node:
            print(f"Node name: {node.name}, OpType: {node.op_type}")

class Saver:
    @classmethod
    def save(cls,model: onnx.ModelProto, path_to_save: str = "modif_onnx.onnx"):
        onnx.save(model, path_to_save)
        print(f"Model saved to {path_to_save}")

class Constructor:
    def find_output(model:onnx.ModelProto,conv_node_name:str):
        conv_output_name = None
        for node in model.graph.node:
            if node.name == conv_node_name:
                conv_output_name = node.output[0]
                break
        return conv_output_name
    @classmethod
    def __add_sigmoid(cls,model:onnx.ModelProto, node_name:str, node_output:str) -> onnx.ModelProto:
        conv_output_name = cls.find_output(model,node_name)
        sigmoid_node = helper.make_node(
            op_type="Sigmoid", 
            inputs=[conv_output_name], 
            outputs=[node_output]
        )

        return sigmoid_node, node_output
    @classmethod
    def ___add_reduce_sum(cls,node_name:str,reduce_sum_output_name:str) -> onnx.ModelProto:
        
        reduce_sum_node = helper.make_node(
            op_type="ReduceSum",  
            inputs=[node_name],  
            outputs=[reduce_sum_output_name], 
            name="ReduceSum",  
            axes=[1],
            keepdims=1  
        )
        return reduce_sum_node,reduce_sum_output_name
    @classmethod
    def __add_clip(cls,node_name:str, clip_output_name :str) -> onnx.ModelProto:
        clip_node = helper.make_node(
            op_type="Clip",  
            inputs=[node_name],  
            outputs=[clip_output_name],  
            name="Clip_365",
            min=0.0,
            max=1.0   
        )

        return clip_node,clip_output_name
    
    @classmethod
    def __add_output_node(cls,model:onnx.ModelProto, node_dict:dict) -> onnx.ModelProto:
        """
        The method to add output to node
        """
        output_node = node_dict["name"]
        output_shape = node_dict["shape"]
        new_output = helper.make_tensor_value_info(
            name = output_node,
            elem_type=TensorProto.FLOAT,
            shape=output_shape
        )

        model.graph.output.append(new_output)

        if not any(tensor.name == output_node for tensor in model.graph.value_info):
            value_info = helper.make_tensor_value_info(
                name=output_node,
                elem_type=TensorProto.FLOAT,
                shape=output_shape
            )
            model.graph.value_info.append(value_info)
        
        return model
    
    @classmethod
    def create_layers(cls,model:onnx.ModelProto,group_node:str) -> onnx.ModelProto:
        """
        add model sigmoid, reduce_sum, clip
        """

        for id, nodes in enumerate(group_node):
            
            if (id+1)%2 == 0:
                model = cls.__add_output_node(model, nodes[0])
            else:
                
                sigmoid_node,sigmoid_output_name = cls.__add_sigmoid(model,nodes[0],nodes[1])
                reduce_sum_node,reduce_sum_output_name = cls.___add_reduce_sum(sigmoid_output_name,nodes[2])
                clip_node,clip_output_name = cls.__add_clip(reduce_sum_output_name, nodes[-3])

                model.graph.node.extend([sigmoid_node, reduce_sum_node, clip_node])
                dict_sigmoid_output = nodes[-2] 
                dict_sigmoid_output["name"] = nodes[1]
                model = cls.__add_output_node(model, dict_sigmoid_output)

                dict_clip_output = nodes[-1]
                dict_clip_output["name"] = clip_output_name
                model = cls.__add_output_node(model, dict_clip_output)
                

        return model

class Deleter_nodes:
    nodes_to_delete = ["/model.22/Concat_5","/model.22/Sigmoid",
                       "/model.22/Mul_2","/model.22/Concat_4","/model.22/Div_1",
                       "/model.22/Add_2","/model.22/Sub_1","/model.22/Add_1","/model.22/Sub",
                       "/model.22/Slice","/model.22/Slice_1","/model.22/dfl/Reshape_1",
                       "/model.22/dfl/conv/Conv","/model.22/dfl/Softmax","/model.22/dfl/Transpose",
                       "/model.22/dfl/Reshape","/model.22/Concat_3","/model.22/Split",
                       "/model.22/Reshape_2","/model.22/Concat_2","/model.22/Reshape","/model.22/Concat",
                       "/model.22/Reshape_1","/model.22/Concat_1"
                       ]
    output_tensor_delete = "output0"

    # name nodes: [default layer name, sigmoid name, reduce_sum_name, reduce_sum_name_output, clip_name,output_name],  
    # brach near: [default layer name,output_name] 

    start_nodes_for_add = [["/model.22/cv3.2/cv3.2.2/Conv","onnx::ReduceSum_365","/model.22/ReduceSum_2_output_0","onnx::ReduceSum_365","/model.22/Clip_2","369",
                            {"shape":[1, 80, 20, 20]}, {"shape":[1,1,20,20]}], 
                           [{"name":"/model.22/cv2.2/cv2.2.2/Conv_output_0",
                             "shape":[1,64,20,20]}],
                             [
                              "/model.22/cv3.1/cv3.1.2/Conv","onnx::ReduceSum_346","/model.22/ReduceSum_1_output_0","onnx::ReduceSum_346","/model.22/Clip_1","350",
                              {"shape":[1, 80, 40, 40]}, {"shape":[1,1,40,40]}
                             ],
                             [{"name":"/model.22/cv2.1/cv2.1.2/Conv_output_0",
                              "shape":[1,64,20,20]}]
                          ]
                    
    #check by name inside https://netron.app/
    # start_node_for_add = "/model.22/cv3.2/cv3.2.2/Conv"

    def save(model) -> None:
        Saver.save(model)
    @classmethod
    def delete_useless_output(cls,model:onnx.ModelProto) -> onnx.ModelProto:
        output_to_remove = None
        for output in model.graph.output:
            if output.name == cls.output_tensor_delete:
                output_to_remove = output
                break

        # Если выходной тензор найден, удаляем его
        if output_to_remove is not None:
            model.graph.output.remove(output_to_remove)
            print(f"Output tensor '{cls.output_tensor_delete}' delete.")
        else:
            print(f"Output tensor with name: '{cls.output_tensor_delete}' not found.")
        return model
    def delete_useless_node(model:onnx.ModelProto) -> onnx.ModelProto:
        used_tensors = set()
        for node in model.graph.node:
            used_tensors.update(node.input)
            used_tensors.update(node.output)

        # clean ValueInfoProto for not connected nodes
        initializers = {init.name for init in model.graph.initializer}
        inputs = {inp.name for inp in model.graph.input}
        outputs = {out.name for out in model.graph.output}

        value_info_to_remove = [
            vi for vi in model.graph.value_info
            if vi.name not in used_tensors and vi.name not in initializers and vi.name not in inputs and vi.name not in outputs
        ]

        for vi in value_info_to_remove:
            model.graph.value_info.remove(vi)

        return model
    
    # def make_output(model:onnx.ModelProto) -> onnx.ModelProto:
    #     #find output tensor
    #     new_output_name = None
    #     for node in reversed(model.graph.node):
    #         if node.output:
    #             new_output_name = node.output[0]
    #             break

    #     if new_output_name is None:
    #         raise ValueError("Не удалось найти новый выходной тензор после удаления узлов.")

    #     # clear old output
    #     model.graph.output[:] = [] 

    #     # make new output
    #     new_output = onnx.helper.ValueInfoProto()
    #     new_output.name = new_output_name
    #     model.graph.output.extend([new_output])

    #     return model

    @classmethod
    def add_nodes(cls,model:onnx.ModelProto):
        model = Constructor.create_layers(model,cls.start_nodes_for_add)
        return model

    @classmethod
    def delete(cls,model:onnx.ModelProto):
        nodes_to_remove = [node for node in model.graph.node if node.name in cls.nodes_to_delete]

        # check what nodes exist
        if len(nodes_to_remove) != len(cls.nodes_to_delete):
            existing_node_names = {node.name for node in model.graph.node}
            missing_nodes = set(cls.nodes_to_delete) - existing_node_names
            raise ValueError(f"Didn't find nodes: {missing_nodes}")
        
        # delete node 
        for node in nodes_to_remove:
            model.graph.node.remove(node)
            print(f"Delete node: {node.name}")

        #delete useless nodes
        model = cls.delete_useless_node(model)
        model = cls.delete_useless_output(model)
        model = cls.add_nodes(model)
        # model = cls.make_output(model)
        model = cls.save(model)

            

if __name__ == "__main__":
    model_pth = WAY_TO_YOLOV8 + MODEL_TYPE
    model = onnx.load(model_pth)
    # Informator.get_info(model)
    Deleter_nodes.delete(model)






