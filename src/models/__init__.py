from tensorflow import keras

def save_models(model, prefix):
    model['ae'].save(prefix + '-ae.h5')
    model['enc'].save(prefix + '-enc.h5')
    model['dec'].save(prefix + '-dec.h5')
    print("Models saved to " + prefix + " ...")

def load_models(prefix):
    model = dict()
    for k in ['ae', 'enc', 'dec']:
        fname = prefix + '-' + k + '.h5'
        model[k] = keras.models.load_model(fname)
    return model

# custom models cannot be saved by the above methods, we can only save weights

def save_models_weights(model, prefix):
    model['ae'].save_weights(prefix + '-ae.h5w')
    model['enc'].save_weights(prefix + '-enc.h5w')
    model['dec'].save_weights(prefix + '-dec.h5w')
    print("Models weights saved to " + prefix + " ...")
    
def load_models_weights(model, prefix):
    model['ae'].load_weights(prefix + '-ae.h5w')
    model['enc'].load_weights(prefix + '-enc.h5w')
    model['dec'].load_weights(prefix + '-dec.h5w')