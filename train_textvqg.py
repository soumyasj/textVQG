#Training script for textVQG

from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
import argparse
import json
import logging
import os
import random
import time
import torch
import torch.nn as nn
from utils import NLGEval
from models import textVQG
from utils import Vocabulary
from utils import get_glove_embedding
from utils import get_loader
from utils import load_vocab
from utils import process_lengths
from utils import gaussian_KL_loss



def create_model(args, vocab, embedding=None):
    
    # Load GloVe embedding.
    if args.use_glove:
        embedding = get_glove_embedding(args.embedding_name,
                                        args.hidden_size,
                                        vocab)
    else:
        embedding = None

    # Build the models
    logging.info('Creating textVQG model...')
    vqg = textVQG(len(vocab), args.max_length, args.hidden_size,
             vocab(vocab.SYM_SOQ), vocab(vocab.SYM_EOS),
             num_layers=args.num_layers,
             rnn_cell=args.rnn_cell,
             dropout_p=args.dropout_p,
             input_dropout_p=args.input_dropout_p,
             encoder_max_len=args.encoder_max_len,
             embedding=embedding,
             num_att_layers=args.num_att_layers,
             z_size=args.z_size)
    return vqg


def evaluate(vqg, data_loader, criterion, l2_criterion, args):
    
    vqg.eval()
    
    total_steps = len(data_loader)
    if args.eval_steps is not None:
        total_steps = min(len(data_loader), args.eval_steps)
    start_time = time.time()
    total_gen_loss = 0.0
    total_kl = 0.0
    total_z_t_kl = 0.0
    for iterations, (images, questions, answers,
           qindices) in enumerate(data_loader):

        # Set mini-batch dataset
        if torch.cuda.is_available():
            images = images.cuda()
            questions = questions.cuda()
            answers = answers.cuda()
            qindices = qindices.cuda()
        alengths = process_lengths(answers)

        # Forward, Backward and Optimize
        image_features = vqg.encode_images(images)
        answer_features = vqg.encode_answers(answers, alengths)
        mus, logvars = vqg.encode_into_z(image_features, answer_features)
        zs = vqg.reparameterize(mus, logvars)
        (outputs, _, other) = vqg.decode_questions(
                image_features, zs, questions=questions,
                teacher_forcing_ratio=1.0)

        # Reorder the questions based on length.
        questions = torch.index_select(questions, 0, qindices)

        # Ignoring the start token.
        questions = questions[:, 1:]
        qlengths = process_lengths(questions)

        # Convert the output from MAX_LEN list of (BATCH x VOCAB) ->
        # (BATCH x MAX_LEN x VOCAB).
        outputs = [o.unsqueeze(1) for o in outputs]
        outputs = torch.cat(outputs, dim=1)
        outputs = torch.index_select(outputs, 0, qindices)

        # Calculate the loss.
        targets = pack_padded_sequence(questions, qlengths,
                                       batch_first=True)[0]
        outputs = pack_padded_sequence(outputs, qlengths,
                                       batch_first=True)[0]
        gen_loss = criterion(outputs, targets)
        total_gen_loss += gen_loss.data.item()

        # Get KL loss if it exists.
        kl_loss = gaussian_KL_loss(mus, logvars)
        total_kl += kl_loss.item()

        


        # Quit after eval_steps.
        if args.eval_steps is not None and iterations >= args.eval_steps:
            break

        # Print logs
        if iterations % args.log_step == 0:
             delta_time = time.time() - start_time
             start_time = time.time()
             logging.info('Time: %.4f, Step [%d/%d], gen loss: %.4f, '
                          'KL: %.4f '
                          'z-t-KL: %.4f'
                         % (delta_time, iterations, total_steps,
                            total_gen_loss/(iterations+1),
                            total_kl/(iterations+1)))
    
    return total_gen_loss / (iterations+1)


def run_eval(vqg, data_loader, criterion, l2_criterion, args, epoch,
             scheduler, info_scheduler):
    logging.info('=' * 80)
    start_time = time.time()
    val_gen_loss, val_info_loss = evaluate(
            vqg, data_loader, criterion, l2_criterion, args)
    delta_time = time.time() - start_time
    scheduler.step(val_gen_loss)
    scheduler.step(val_info_loss)
    logging.info('Time: %.4f, Epoch [%d/%d], Val-gen-loss: %.4f, '
                 'Val-info-loss: %.4f' % (
        delta_time, epoch, args.num_epochs, val_gen_loss, val_info_loss))
    logging.info('=' * 80)





def compare_outputs(images, questions, answers, 
                    alengths, vqg, vocab, logging, cat2name,
                    args, num_show=5):
    
    vqg.eval()

    # Forward pass through the model.
    outputs = vqg.predict_from_answer(images, answers, lengths=alengths)

    for _ in range(num_show):
        logging.info("         ")
        i = random.randint(0, images.size(0) - 1)  # Inclusive.

        

        # Log the outputs.
        output = vocab.tokens_to_words(outputs[i])
        question = vocab.tokens_to_words(questions[i])
        answer = vocab.tokens_to_words(answers[i])
        logging.info('Sampled question : %s\n'
                     'Target  question (%s): %s -> %s'
                     % (output,
                        question, answer))
        logging.info("         ")


def compute_two_gaussian_loss(mu1, logvar1, mu2, logvar2):
    
    numerator = logvar1.exp() + torch.pow(mu1 - mu2, 2)
    fraction = torch.div(numerator, (logvar2.exp() + 1e-8))
    kl = 0.5 * torch.sum(logvar2 - logvar1 + fraction - 1)
    return kl / (mu1.size(0) + 1e-8)


def train(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # Save the arguments.
    with open(os.path.join(args.model_path, 'args.json'), 'w') as args_file:
        json.dump(args.__dict__, args_file)

    # Config logging.
    log_format = '%(levelname)-8s %(message)s'
    logfile = os.path.join(args.model_path, 'train1.log')
    logging.basicConfig(filename=logfile, level=logging.INFO, format=log_format)
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info(json.dumps(args.__dict__))

    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(args.crop_size,
                                     scale=(1.00, 1.2),
                                     ratio=(0.75, 1.3333333333333333)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    # Load vocabulary wrapper.
    vocab = load_vocab(args.vocab_path)

    

    # Build data loader
    logging.info("Building data loader...")
    train_sampler = None
    val_sampler = None
    if os.path.exists(args.train_dataset_weights):
        train_weights = json.load(open(args.train_dataset_weights))
        train_weights = torch.DoubleTensor(train_weights)
        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(
                train_weights, len(train_weights))
    if os.path.exists(args.val_dataset_weights):
        val_weights = json.load(open(args.val_dataset_weights))
        val_weights = torch.DoubleTensor(val_weights)
        val_sampler = torch.utils.data.sampler.WeightedRandomSampler(
                val_weights, len(val_weights))
    data_loader = get_loader(args.dataset, transform,
                                 args.batch_size, shuffle=False,
                                 num_workers=args.num_workers,
                                 max_examples=args.max_examples,
                                 sampler=train_sampler)
    val_data_loader = get_loader(args.val_dataset, transform,
                                     args.batch_size, shuffle=False,
                                     num_workers=args.num_workers,
                                     max_examples=args.max_examples,
                                     sampler=val_sampler)
    logging.info("Done")

    vqg = create_model(args, vocab)
    if args.load_model is not None:
        vqg.load_state_dict(torch.load(args.load_model))
    logging.info("Done")

    # Loss criterion.
    pad = vocab(vocab.SYM_PAD)  # Set loss weight for 'pad' symbol to 0
    criterion = nn.CrossEntropyLoss(ignore_index=pad)
    l2_criterion = nn.MSELoss()

    # Setup GPUs.
    if torch.cuda.is_available():
        logging.info("Using available GPU...")
        vqg.cuda()
        criterion.cuda()
        l2_criterion.cuda()

    # Parameters to train.
    gen_params = vqg.generator_parameters()
    info_params = vqg.info_parameters()
    learning_rate = args.learning_rate
    info_learning_rate = args.info_learning_rate
    gen_optimizer = torch.optim.Adam(gen_params, lr=learning_rate)
    info_optimizer = torch.optim.Adam(info_params, lr=info_learning_rate)
    scheduler = ReduceLROnPlateau(optimizer=gen_optimizer, mode='min',
                                  factor=0.1, patience=args.patience,
                                  verbose=True, min_lr=1e-7)
    info_scheduler = ReduceLROnPlateau(optimizer=info_optimizer, mode='min',
                                       factor=0.1, patience=args.patience,
                                       verbose=True, min_lr=1e-7)

    # Train the model.
    total_steps = len(data_loader)
    start_time = time.time()
    n_steps = 0

    # Optional losses. Initialized here for logging.
    
    kl_loss = 0.0
    
    for epoch in range(args.num_epochs):
        for i, (images, questions, answers,
                 qindices) in enumerate(data_loader):
            n_steps += 1

            # Set mini-batch dataset.
            if torch.cuda.is_available():
                images = images.cuda()
                questions = questions.cuda()
                answers = answers.cuda()
               
                qindices = qindices.cuda()
            alengths = process_lengths(answers)

            # Eval now.
            if (args.eval_every_n_steps is not None and
                    n_steps >= args.eval_every_n_steps and
                    n_steps % args.eval_every_n_steps == 0):
                run_eval(vqg, val_data_loader, criterion, l2_criterion,
                         args, epoch, scheduler, info_scheduler)
                compare_outputs(images, questions, answers,
                                alengths, vqg, vocab, logging, cat2name, args)

            # Forward.
            vqg.train()
            gen_optimizer.zero_grad()
            info_optimizer.zero_grad()
            image_features = vqg.encode_images(images)
            answer_features = vqg.encode_answers(answers, alengths)

            # Question generation.
            mus, logvars = vqg.encode_into_z(image_features, answer_features)
            zs = vqg.reparameterize(mus, logvars)
            (outputs, _, _) = vqg.decode_questions(
                    image_features, zs, questions=questions,
                    teacher_forcing_ratio=1.0)

            # Reorder the questions based on length.
            questions = torch.index_select(questions, 0, qindices)

            # Ignoring the start token.
            questions = questions[:, 1:]
            qlengths = process_lengths(questions)

            # Convert the output from MAX_LEN list of (BATCH x VOCAB) ->
            # (BATCH x MAX_LEN x VOCAB).
            outputs = [o.unsqueeze(1) for o in outputs]
            outputs = torch.cat(outputs, dim=1)
            outputs = torch.index_select(outputs, 0, qindices)

            # Calculate the generation loss.
            targets = pack_padded_sequence(questions, qlengths,
                                           batch_first=True)[0]
            outputs = pack_padded_sequence(outputs, qlengths,
                                           batch_first=True)[0]
            gen_loss = criterion(outputs, targets)
            total_loss = 0.0
            total_loss += args.lambda_gen * gen_loss
            gen_loss = gen_loss.item()

            # Variational loss.
            kl_loss = gaussian_KL_loss(mus, logvars)
            total_loss += args.lambda_z * kl_loss
            kl_loss = kl_loss.item()

            

            # Generator Backprop.
            total_loss.backward()
            gen_optimizer.step()

            


            # Print log info
            if i % args.log_step == 0:
                delta_time = time.time() - start_time
                start_time = time.time()
                logging.info('Time: %.4f, Epoch [%d/%d], Step [%d/%d], '
                             'LR: %f,  KL: %.4f, '
                             % (delta_time, epoch, args.num_epochs, i,
                                total_steps, gen_optimizer.param_groups[0]['lr'],
                                 kl_loss))

            

            # Save the models
            if args.save_step is not None and (i+1) % args.save_step == 0:
                torch.save(vqg.state_dict(),
                           os.path.join(args.model_path,
                                        'vqg-tf-%d-%d.pkl'
                                        % (epoch + 1, i + 1)))

        torch.save(vqg.state_dict(),
                   os.path.join(args.model_path,
                                'vqg-tf-%d.pkl' % (epoch+1)))

        # Evaluation and learning rate updates.
        run_eval(vqg, val_data_loader, criterion, l2_criterion,
                 args, epoch, scheduler, info_scheduler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data parameters.
    parser.add_argument('--vocab-path', type=str,
                        default='/home/shankar/Desktop/VQA_REU/Stvqa/vocab_iq_2.json',
                        help='Path for vocabulary wrapper.')
    parser.add_argument('--dataset', type=str,
                        default='/home/shankar/Desktop/VQA_REU/Stvqa/iq_dataset_cat12.hdf5',
                        help='Path for train annotation json file.')
    parser.add_argument('--val-dataset', type=str,
                        default='/home/shankar/Desktop/VQA_REU/Stvqa/iq_dataset_cat12.hdf5',
                        help='Path for train annotation json file.')
    parser.add_argument('--train-dataset-weights', type=str,
                        default='/home/shankar/Desktop/VQA_REU/Stvqa/data/processed/iq_train_dataset_weights2.json',
                        help='Location of sampling weights for training set.')
    parser.add_argument('--val-dataset-weights', type=str,
                        default='/home/shankar/Desktop/VQA_REU/Stvqa/data/processed/iq_val_dataset_weights2.json',
                        help='Location of sampling weights for training set.')
    parser.add_argument('--load-model', type=str, default=None,
                        help='Location of where the model weights are.')

    # Session parameters.
    parser.add_argument('--model-path', type=str, default='weights2/tf1/',
                        help='Path for saving trained models')
    parser.add_argument('--crop-size', type=int, default=224,
                        help='Size for randomly cropping images')
    parser.add_argument('--log-step', type=int, default=10,
                        help='Step size for prining log info')
    parser.add_argument('--save-step', type=int, default=None,
                        help='Step size for saving trained models')
    parser.add_argument('--eval-steps', type=int, default=100,
                        help='Number of eval steps to run.')
    parser.add_argument('--eval-every-n-steps', type=int, default=300,
                        help='Run eval after every N steps.')
    parser.add_argument('--num-epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--info-learning-rate', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--max-examples', type=int, default=None,
                        help='For debugging. Limit examples in database.')
    # Model parameters
    parser.add_argument('--rnn-cell', type=str, default='LSTM',
                        help='Type of rnn cell (GRU, RNN or LSTM).')
    parser.add_argument('--hidden-size', type=int, default=512,
                        help='Dimension of lstm hidden states.')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='Number of layers in lstm.')
    parser.add_argument('--max-length', type=int, default=20,
                        help='Maximum sequence length for outputs.')
    parser.add_argument('--encoder-max-len', type=int, default=4,
                        help='Maximum sequence length for inputs.')
    parser.add_argument('--bidirectional', action='store_true', default=False,
                        help='Boolean whether the RNN is bidirectional.')
    parser.add_argument('--use-glove', action='store_true',
                        help='Whether to use GloVe embeddings.')
    parser.add_argument('--embedding-name', type=str, default='6B',
                        help='Name of the GloVe embedding to use.')
    parser.add_argument('--dropout-p', type=float, default=0.3,
                        help='Dropout applied to the RNN model.')
    parser.add_argument('--input-dropout-p', type=float, default=0.3,
                        help='Dropout applied to inputs of the RNN.')
    parser.add_argument('--num-att-layers', type=int, default=2,
                        help='Number of attention layers.')
    parser.add_argument('--z-size', type=int, default=512,
                        help='Dimensions to use for hidden variational space.')

    # Lambda values.
    parser.add_argument('--lambda-z', type=float, default=0.001,
                        help='coefficient to be added in front of the kl loss.')
    

    args = parser.parse_args()
    train(args)
    Vocabulary()
