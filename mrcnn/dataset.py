from mrcnn import utils
import numpy as np
import os


class CVPRDataset(utils.Dataset):
    def load_examples(self, images_dir, labels_dir, example_ids, class_dict):
        """
        Fills out self.class_info list with corresponding dictionaries
        for each mask.
        Fills out self.image_info list with corresponding dictionaries
        for each image in "images_dir".
        """
        # Add Classes
        self.class_info[0]["source"] = "CVPR_Dataset"
        class_ids = list(class_dict.keys())
        class_names = list(class_dict.values())    
        for i in range(len(class_ids)):
            self.add_class("CVPR_Dataset", class_ids[i], class_names[i])
        
        # Add Images
        #image_ids = os.listdir(images_dir)
        for example_id in example_ids:
            image_id = example_id + ".jpg"
            
            # derive example_id, image_path, mask_path from image_id and directories
            #example_id = image_id[:-4]
            image_path = os.path.join(images_dir, image_id)
            mask_path = os.path.join(labels_dir, example_id + '_instanceIds.png')
            
            # add image info
            self.add_image("CVPR_Dataset", image_id = example_id, path = image_path,
                           mask_path = mask_path)
        print(len(example_ids))
        pass
    
    def load_image(self, image_id):
        """
        Load the specified image and return a [H,W,3] Numpy array.
        
        Arguments:
        image_id -- the datasets internal image id
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image
    
    def load_mask(self, image_id):
        """
        Arguments:
        image_id -- the datasets internal image id
        
        Returns:
        masks - A bool array of shape [height, width, instance count] with
                a binary mask per instance.
        class_ids - a 1D array of (internal) class IDs of the instance masks.
        """
        
        info = self.image_info[image_id]
        source = info["source"]
        
        # load the mask corresponding with image_id
        cvpr_mask = skimage.io.imread(info['mask_path'])
        
        # compute instances, class_ids
        instances = np.unique(cvpr_mask)
        class_ids = np.array(instances / 1000, dtype = np.int16)
        
        # convert external class ids to internal. If not in self.class_ids, assign number higher than
        # highest inetenal index
        def external_to_internal_class_id(external_id, source):
            try: class_id = self.class_from_source_map["{}.{}".format(source, str(external_id))]
            except: class_id = self.num_classes + 1
            return class_id
        class_ids = np.array([external_to_internal_class_id(class_id, source) for class_id in class_ids])
    
        # filter out "0" (background) and any class_ids that are not in class_dict
        inds = []
        for i, class_id in enumerate(class_ids):
            if class_id != 0 and class_id in self.class_ids:
                inds.append(i) 
        instances = instances[inds]
        class_ids = class_ids[inds]
        
        # initialize boolean mask with 0 values and shape [Height, Width, instances]
        mask = np.zeros((cvpr_mask.shape[0], cvpr_mask.shape[1], len(instances)))
        
        # loop through instances, insert each instance into third dimension of mask (mask[:, :, i])
        for i, instance in enumerate(instances):
            instance_mask = np.array(cvpr_mask == instance, dtype = np.uint8)
            mask[:, :, i] = instance_mask

        #return mask, class_ids
        return mask.astype(np.bool), class_ids.astype(np.int32)




