# Creating-Pokemon-With-DCGAN

Unique creatures known as Pokémon take on various shapes, colours, and sizes. Recently, there are now over 1,000 different Pokémon within the Nintendo franchise. I am always impressed by how they are able to come up with so many different designs. Using a deep convolutional generative adversarial network, I wanted to see if I could create a Pokémon. With an input of over 1,600 images, I trained a model to try to output an image of a Pokémon created from the features learned. My goal was to achieve pictures similar enough that it could fool another model into thinking it was actually a real Pokémon. There are multiple modifications and directions I would like to take this project in the future, but for now this is a baseline result that I’m content with.

The input images are inside the data/input folder. The images that were created are inside the data/output folder. The name of the file corresponds to the epoch that image was created. I saved the image every other epoch, so as not to clutter the data too much. I chose seven images that performed relatively well in their respective training lifetime and put them inside the data/"7 Image Progression" folder.
