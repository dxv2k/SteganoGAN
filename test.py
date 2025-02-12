from steganogan import SteganoGAN

# Load a pretrained model (choose from 'basic', 'dense', or 'residual')
model = SteganoGAN.load(architecture='dense')

# Encode a message into an image
model.encode(
    cover='/home/dxv2k/stegnography/Week6/SteganoGAN/cola.jpg',  # Replace with your image path
    output='encoded_output.png',
    text='This is a secret message!'
)

# Decode the message from the encoded image
decoded_message = model.decode('encoded_output.png')
print('Decoded message:', decoded_message)

# /home/dxv2k/stegnography/Week6/SteganoGAN/cola.jpg
