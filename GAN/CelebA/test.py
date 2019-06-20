import pickle
import PIL.Image
from GAN.CelebA.loss  import *
import GAN.CelebA.tfutil as tfutil
import numpy as np
import tensorflow as tf


# Initialize TensorFlow session.
tf.InteractiveSession()


# Import official CelebA-HQ networks.
with open('network-snapshot-008640.pkl', 'rb') as file:
    G, D, Gs = pickle.load(file)


# Generate latent vectors.
latents = np.random.RandomState(400).randn(1000, *Gs.input_shapes[0][1:]) # 1000 random latents

if False:
	latents = latents[[477, 56, 83, 887, 583, 391, 86, 340, 341, 415]] # hand-picked top-10

else:
	latents = latents[[83, 887]]
	print(np.linalg.norm(latents[1]-latents[0]))
	theta = np.linspace(0.0,1.0,num=32)
	newlatents = [(latents[0]*(1-theta[i]) + latents[1]*theta[i]) for i in range(np.shape(theta)[0])]
	latents = np.asarray(newlatents,dtype=np.float32)

lbls =  np.zeros([latents.shape[0]] + Gs.input_shapes[1][1:])


plchldr = G.input_templates[0]


print("plchldr=")
print(plchldr)

lbls_plchldr = G.input_templates[1]

plchldr_new = tf.placeholder(shape=(None,512), dtype='float32')
gnrtd_new = G.get_output_for(plchldr_new, lbls_plchldr, is_training=False)
fake_scores_out_new, fake_labels_out_new = fp32(D.get_output_for(gnrtd_new, is_training=False))
imgs_new =  tf.get_default_session().run( gnrtd_new, feed_dict={plchldr_new : latents, lbls_plchldr :lbls })




coefficients= tf.Variable(np.ones((4*512),dtype=np.float32), name= "coeffficient")


curve = plchldr*tf.reduce_sum(coefficients)

gnrtd = G.get_output_for(curve, lbls_plchldr, is_training=False)
fake_scores_out, fake_labels_out = fp32(D.get_output_for(gnrtd, is_training=False))

grdnt = tf.gradients([fake_scores_out],[coefficients],aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)

print(fake_scores_out.shape)

init = tf.initialize_variables([coefficients])
tf.get_default_session().run(init)

imgs =  tf.get_default_session().run( gnrtd, feed_dict={plchldr : latents, lbls_plchldr :lbls })


diffs = [np.linalg.norm(imgs[i+1] - imgs[i]) for i in range(np.shape(latents)[0]-1)]
print(diffs)
#rslt, imgs,grds =  tf.get_default_session().run([fake_scores_out, gnrtd,grdnt], feed_dict={plchldr : latents, lbls_plchldr :lbls })




imgsPlot = np.clip(np.rint((imgs + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8) # [-1,1] => [0,255]
imgsPlot = imgsPlot.transpose(0, 2, 3, 1) # NCHW => NHWC



for idx in range(imgsPlot.shape[0]):
    PIL.Image.fromarray(imgsPlot[idx], 'RGB').save('test_img%d.png' % idx)

#print(D.list_layers())
#D.print_layers()

#test = tf.get_default_graph().get_tensor_by_name("D_paper/4x4/Conv/weight:0")
#print(test)

test2 = [v for v in tf.global_variables() if v.name == "D/4x4/Conv/weight:0"][0]
test3 = test2[:,:,512,:].assign(tf.zeros((3,3,512)))


a = tf.get_default_session().run( test3, feed_dict={plchldr: latents, lbls_plchldr: lbls} )
#a = tf.get_default_session().run( test3 , feed_dict={plchldr: latents, lbls_plchldr: lbls} )
tfutil.set_vars({test2: a})

gradient = tf.get_default_session().run( grdnt, feed_dict={plchldr: latents, lbls_plchldr: lbls} )
print(np.shape(gradient))

disc_values = D.run(imgs)[0]
print(disc_values)