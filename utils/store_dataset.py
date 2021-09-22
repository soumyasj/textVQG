"""Transform all the Scene Text VQA dataset if nto a hdf5 dataset.
"""

from PIL import Image
from torchvision import transforms

import argparse
import json
import h5py
import numpy as np
import os
import progressbar

from train_utils import Vocabulary
from vvv import load_vocab
from vvv import process_text


def create_answer_mapping(annotations,annos):
    
    data = annotations
    answers = {}
    image_ids = set()
    for q in data:
        question_id = q.get("question_id")
        answer = q.get("answer")
        answers[question_id] = answer
        image_ids.add(q.get("image_id"))
    return answers, image_ids


def save_dataset(image_dir, questions, annotations, vocab,output,
                 im_size=224, max_q_length=20, max_a_length=4,
                 with_answers=False):
    
    # Load the data.
    max_ocr_len=8
    vocab = load_vocab(vocab)
    with open(annotations) as f:
        annos = json.load(f)
    with open(questions) as f:
        questions = json.load(f)

    # Get the mappings from qid to answers.
    qid2ans, image_ids = create_answer_mapping(annos, annos)
    total_questions = len(qid2ans.keys())
    total_images = len(image_ids)
    print ("Number of images to be written: %d" % total_images)
    print ("Number of QAs to be written: %d" % total_questions)

    h5file = h5py.File(output, "w")
    d_questions = h5file.create_dataset(
        "questions", (total_questions, max_q_length), dtype='i')
    d_indices = h5file.create_dataset(
        "image_indices", (total_questions,), dtype='i')
    d_images = h5file.create_dataset(
        "images", (total_images, im_size, im_size, 3), dtype='f')
    d_answers = h5file.create_dataset(
        "answers", (total_questions, max_a_length), dtype='i')
    d_ocr_positions = h5file.create_dataset(
        "ocr_positions", (total_questions, max_ocr_len), dtype='f')
    

    # Create the transforms we want to apply to every image.
    transform = transforms.Compose([
        transforms.Resize((im_size, im_size))])

    # Iterate and save all the questions and images.
    # bar = progressbar.ProgressBar(maxval=total_questions)
    i_index = 1
    q_index = 1
    done_img2idx = {}
    questions1 = questions
    #print(questions1)
    for entry in questions1:
        image_id = entry.get("image_id")
        question_id = entry.get("question_id")
        if image_id not in image_ids:
            continue
        if question_id not in qid2ans:
            continue
        if image_id not in done_img2idx:
            try:
                path =  "D:/NEW_LAPTOP/Desktop_files/Research/check_textvqg/data/train_images/"+ (image_id)+'.jpg'
                image = Image.open(os.path.join(image_dir, path)).convert('RGB')
            except IOError:
                path =  "D:/NEW_LAPTOP/Desktop_files/Research/check_textvqg/data/train_images/"+ (image_id)+'.jpg'
                image = Image.open(os.path.join(image_dir, path)).convert('RGB')
            image = transform(image)
            
            d_images[i_index, :, :, :] = np.array(image)
            done_img2idx[image_id] = i_index
            i_index += 1
        q, length = process_text(entry['question'], vocab,
                                 max_length=max_q_length)
        d_questions[q_index, :length] = q
        # print(q, "----" ,d_questions[q_index, :length],"len is: ", length)
        answer = qid2ans[question_id]
        a, length = process_text(answer, vocab,
                                 max_length=max_a_length)
        d_answers[q_index, :length] = a
        ocr_len = len(list(entry.get("ocr_position").values()))
        process_ocr_pos = entry.get("ocr_position")
        # print("pos: ", list((process_ocr_pos.values())))
        # print(entry.get("ocr_position"))
        d_ocr_positions[q_index, :8] = list(process_ocr_pos.values())
        # print(d_ocr_positions)
        # print("holaaaa", d_ocr_positions[q_index, ocr_len])
        
    
        
        d_indices[q_index] = done_img2idx[image_id]
        q_index += 1
        # bar.update(q_index)
    h5file.close()
    print ("Number of images written: %d" % i_index)
    print ("Number of QAs written: %d" % q_index)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Inputs.
    parser.add_argument('--image-dir', type=str, default='D:/NEW_LAPTOP/Desktop_files/Research/check_textvqg/data/train_images',
                        help='directory for resized images')
    parser.add_argument('--questions', type=str,
                        default='D:/NEW_LAPTOP/Desktop_files/Research/check_textvqg/textVQG/textvqa_qa_ocr_data.json',
                        help='Path for train annotation file.')
    parser.add_argument('--annotations', type=str,
                        default='D:/NEW_LAPTOP/Desktop_files/Research/check_textvqg/textVQG/textvqa_qa_ocr_data.json',
                        help='Path for train annotation file.')
    
    parser.add_argument('--vocab-path', type=str,
                        default='D:/NEW_LAPTOP/Desktop_files/Research/check_textvqg/textVQG/vocab_iq1.json',
                        help='Path for saving vocabulary wrapper.')

    # Outputs.
    parser.add_argument('--output', type=str,
                        default='D:/NEW_LAPTOP/Desktop_files/Research/check_textvqg/textVQG/textvqg_dataset1.hdf5',
                        help='directory for resized images.')
    

    # Hyperparameters.
    parser.add_argument('--im_size', type=int, default=224,
                        help='Size of images.')
    parser.add_argument('--max-q-length', type=int, default=20,
                        help='maximum sequence length for questions.')
    parser.add_argument('--max-a-length', type=int, default=4,
                        help='maximum sequence length for answers.')
    args = parser.parse_args()

    
    save_dataset(args.image_dir, args.questions, args.annotations, args.vocab_path,
                 args.output, im_size=args.im_size,
                 max_q_length=args.max_q_length, max_a_length=args.max_a_length)
    print('Wrote dataset to %s' % args.output)
    # Hack to avoid import errors.
    Vocabulary()
