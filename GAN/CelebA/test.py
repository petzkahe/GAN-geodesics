import pickle
import numpy as np
import tensorflow as tf
import PIL.Image
from CelebA.loss import *

# Initialize TensorFlow session.
tf.InteractiveSession()


# Import official CelebA-HQ networks.
with open('karras2018iclr-celebahq-1024x1024.pkl', 'rb') as file:
    G, D, Gs = pickle.load(file)


# Generate latent vectors.
latents = np.random.RandomState(1000).randn(1000, *Gs.input_shapes[0][1:]) # 1000 random latents
latents = latents[[477, 56, 83, 887, 583, 391, 86, 340, 341, 415]] # hand-picked top-10
latents = latents[0:10]
#normalization = np.sqrt(np.sum(latents**2, axis=1)).reshape(10,1)
#print(normalization)
#latents=latents/normalization


# Generate dummy labels (not used by the official networks).
#labels = np.zeros([latents.shape[0]] + Gs.input_shapes[1][1:])

#plchldr= tf.placeholder(tf.float32, shape=(10,512),name="abc")
lbls =  np.zeros([latents.shape[0]] + Gs.input_shapes[1][1:])


plchldr = G.input_templates[0]
lbls_plchldr = G.input_templates[1]

#lbls_plchldr = tf.placeholder(tf.float32, shape=lbls.shape)

#gnrtd =  G.run_with_plh(plchldr,lbls_plchldr)
gnrtd = G.get_output_for(plchldr, lbls_plchldr, is_training=False)
fake_scores_out, fake_labels_out = fp32(D.get_output_for(gnrtd, is_training=False))


#grdnt = tf.gradients(fake_scores_out,plchldr)




imgs =  tf.get_default_session().run( gnrtd, feed_dict={plchldr : latents, lbls_plchldr :lbls })

#rslt, imgs,grds =  tf.get_default_session().run([fake_scores_out, gnrtd,grdnt], feed_dict={plchldr : latents, lbls_plchldr :lbls })



#print(np.shape(grds))


imgsPlot = np.clip(np.rint((imgs + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8) # [-1,1] => [0,255]
imgsPlot = imgs.transpose(0, 2, 3, 1) # NCHW => NHWC

# Save images as PNG.

for idx in range(imgsPlot.shape[0]):
    PIL.Image.fromarray(imgsPlot[idx], 'RGB').save('test_img%d.png' % idx)

print(D.list_layers())
D.print_layers()

#disc_values = D.run(imgs)
#print(disc_values)


test = tf.get_default_graph().get_tensor_by_name("D_paper/4x4/Conv/weight:0")
print(test)

test2 = [v for v in tf.global_variables() if v.name == "D_paper/4x4/Conv/weight:0"][0]
print(test2)


# Which of the two versions below actually does disable the minibatch standard deviation?

#test3 = test2[:,:,512,:].assign(tf.zeros((3,3,512)))
#a = tf.get_default_session().run( test3, feed_dict={plchldr: latents, lbls_plchldr: lbls} )

#tfutil.set_vars({test2: a})

disc_values = D.run(imgs)
print(disc_values)