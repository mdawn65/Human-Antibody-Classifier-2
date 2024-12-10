# Human-Antibody-Classifier

![image](https://github.com/user-attachments/assets/05840c8c-eeb5-40a4-bd25-23cb062d2cbb)

## Data
,species,sequence
0,human,SQTLSLTCAVSGYSISSSNWWGWIRQPPGKGLEWIGYIYYSGSTYYNPSLKSRVTMSVDTSKNQFSLKLSSVTAVDTAVYYCARMGYYGRRESWYFDLWGRGTLVTVSS
1,human,GGSLRLSCAASGFTFNNYAMDWVRQAPGKGLEWVSSISGRGDGTYYADSVKGRFTISRDNSKNTLYLQMDSLRPEDTAVYYCAKEEWEAFDYWGQGALVTVSS
2,human,SQTLSLTCTVSGGSISSYYWSWIRQPPGKGLEWIGYIYYTGSTNYNPSLTSRVTISVDTSTNQFSLKLSSVTAADTGVYYCARELAAAYNWFDPWGQGALVTVSS
3,human,GESLKISCKGSGYSFTSYWIGWVRQMPGKGLEWMGIIYPGDSDTRYSPSFQGQVTMSADKSISTAYLQWSSLKASDTAMYYCARHRYGDYRDAFDIWGQGTMVTVSS
4,human,GESLKISCKGSGYSFTTYWIGWVRQMPGKGLEWMGIIYPGDSDTRYSPSFQGQVTISADKSISTAYLQWSSLKASDTAMYYCARRFCSSTSCHFDYWGQGTLVTVSS
5,human,SETLSLTCTVSGGSISSYYWSWIRQPPGKGLEWIGYIYYSGSTNYNPSLKSRVTISVDTSKNQFSLKLSSVTAADTAVYYCATAYDYGDYGAFDIWGQGTMVTVSS
6,human,GGSLRLSCAASGFTFSDYYMSWIRQVPGKGLEWVSYISSTISSSGRSIHYADSVKGRFTISRDNAKNSLFLQMNSLRAEDTAVYYCARGRYYDSIMVYWGQGTLVTVSS
7,human,SETLSLTCTVSGGSISSSSYYWGWIRQPPGKGLEWIGSIYYSGSTYYNPSLKSRVTISVDTSKNQFSLKLSSVTAADTAVYYCARLGGSYRIDYYYMDVWGKGTTVTVSS
8,human,GGSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAFIRYDGSNKYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCATPASIYDSSGYYYYYYYGMDVWGQGTTVTVSS
9,human,SETLSLTCTVSGGSISSGGYYWSWIRQHPGKGLEWIGYIYYSGSTYYNPSLKSRVTISVDTSKNQFSLKLSSVTAADTAVYYCARGGYSYGYMVDYWGQGTLVTVSS
10,human,SETLSLTCTVSGGSISSSSYYWGWIRQPPGKGLEWIGSIYYSGSTYYNPSLKSRVTISVDTSKNQFSLKLSSVTAADTAVYYCARGYSSGWYVVDYWGQGTLVTVSS
11,human,SETLSLTCAVSGGSISSSNWWSWVRQPPGKGLEWIGEIYHSGSTNYNPSLKSRVTISVDKSKNQFSLKLSSVTAADTAVYYCARDNGGVAVAGFDYWGQGTLVTVSS
12,human,GGSLRLSCAASGFTFSSYSMNWVRQAPGKGLEWVSYISSSSSTIYYADSVKGRFTISRDNAKNSLYLQMNSLRDEDTAVYYCARDIPGYYYYYGMDVWGQGTTVTVSS
13,human,GGSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAFIRYDGSNKYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDSGYSSGWYSKQGPPDIWGQGTMVTVSS
...

## Reflections
This project was very fun, and I really enjoyed coding up a LSTM model from scratch. Lots of learning moments and searching online, but this experience taught me perserverence and patience. 

## Challenges I faced

### Hardware
Some challenges I faced was that running the model solely on my computer -- and just simply using my own CPU -- was incredibly slow (each epoch took somewhere around 3.5-4+ hours. And there were intially 10 epochs, now reduced to 7 (will touch on this later)). 

Then, I changed to using Google Colab, but the size of my files and my compute were way to large for the Colab to run, so it kept crashing, even before it got to training my model (simply at the data process step, the Google Colab would crash).

Finally, I asked around and people suggested me to use GPUs, and luckily, my club has a server. Thus, I had to shift to using my club's server, which took me some time to adjust to. Besides learning how to navigate and use the server, the part that I had most difficulty with was using the server's GPU's. Apparently, I hadn't installed Tensorflow correctly with the GPU; there was a lot of searching online and trying out different commands, but after a long time of trial and error (and patience), I was able to connect to the GPU (this [link]([url](https://www.tensorflow.org/install/pip)) is so helpful, follow all of the steps) which ran so much faster (each epoch was 40-50 mins). 

### Software
I did a lot of digging into and experimentation with the creation of a classification model, such as using an LSTM, what layers to use, the architecture, how many epochs were needed, and more. More in the "What I Learned" section below. 

## What I Learned

#### Data Processing
1. Encode the labels (character strings -> integers)
2. Pad the sequences (so they are all of the same length)
3. Split the training and testing data

#### Model
* Why a Sequential model? It's simple and works well for models where layers follow a straightforward, linear structure, perfect for this classification task.
* Embedding layer: Converts input (integer-encoded amino acids) into embeddings (dense vectors of fixed size)
* Why LSTM? LSTM layers are good for learning patterns in sequences.
* Dense layer: Adds a fully connected feedforward layer to further process outputs, allows for model to learn non-linear patterns
* ReLU: Rectified linear unit activation function, applied to introduce non-linearity and prevent vanishing gradients.
* Softmax: Converts the outputs into probabilities that sum to 1 across all classes

#### Extraneous
- Use tmux on the server - it'll allow you to leave the terminal/get off WiFi and your session will continue to run in the background.
- Log your information, it'll be easier to track everything in real-time (and later on)
