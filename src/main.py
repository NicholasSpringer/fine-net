import tensorflow as tf

def train(model, optimizer, train_a, train_p, train_n, batch_size):
    n_inputs = train_a.shape[0]
    n_batches = -(-n_inputs // model.batch_size)
    for i in range(n_batches):
        start = i*model.batch_size
        end = (i+1)*model.batch_size
        batch_a = train_a[start:end]
        batch_p = train_p[start:end]
        batch_n = train_n[start:end]

        with tf.GradientTape() as tape:
            embed_a = model(batch_a)
            embed_p = model(batch_p)
            embed_n = model(batch_n)
            loss = model.loss_function(embed_a, embed_p, embed_n)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    