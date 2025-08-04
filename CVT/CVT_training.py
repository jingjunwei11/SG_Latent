import argparse
import json
import pickle
import time
import shutil
import os
import queue
import numpy as np

import torch
from torch import Tensor
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchtext.data import Dataset

from CVT.CVT_prediction import pre_validate_on_data
from CVT.Conv_model import build_model

from CVT.batch import Batch
from discriminator_Data import DataClassifierLayers
from helpers import load_config, log_cfg, load_checkpoint, make_model_dir, \
    make_logger, set_seed, symlink_update, ConfigurationError, get_latest_checkpoint, dtw_distance, DTWLoss
from model import Model
from loss import RegLoss, HuberLoss
from data import load_data
from data_operate.dataset import make_data_iter
from builders import build_optimizer, build_scheduler, \
    build_gradient_clipper
from constants import TARGET_PAD
from data_operate.dataset import collate_fn

from plot_videos import plot_video, alter_DTW_timing


class CVTTrainManager:

    def __init__(self, model: Model, disc: DataClassifierLayers, config: dict, test=False, ckpt=None) -> None:

        train_config = config["training"]
        model_dir = train_config["model_dir"]
        # If model continue, continues model from the latest checkpoint
        model_continue = train_config.get("continue", True)
        # If the directory has not been created, can't continue from anything
        if not os.path.isdir(model_dir):
            model_continue = False
        if test:
            model_continue = True

        # files for logging and storing
        self.ckpt = ckpt
        self.model_dir = make_model_dir(train_config["model_dir"],
                                        overwrite=train_config.get("overwrite", False),
                                        model_continue=model_continue)
        # Build logger
        self.logger = make_logger(model_dir=self.model_dir)
        self.logging_freq = train_config.get("logging_freq", 250)
        # Build validation files
        self.valid_report_file = "{}/validations.txt".format(self.model_dir)
        self.tb_writer = SummaryWriter(log_dir=self.model_dir + "/tensorboard/")

        # model
        self.model = model

        self.pad_index = 1
        # self.bos_index = self.model.bos_index
        self._log_parameters_list()
        self.target_pad = TARGET_PAD

        # New Regression loss - depending on config
        # self.loss = RegLoss(cfg=config,
        #                     target_pad=self.target_pad)
        # self.mulit_loss = HuberLoss(cfg=config,
        #                     target_pad=self.target_pad)
        self.loss = torch.nn.MSELoss()
        self.mulit_loss = torch.nn.HuberLoss()

        self.normalization = "batch"

        self.disc = disc
        self.disc_opt = build_optimizer(config=config["training"]["disc"], parameters=self.disc.parameters())

        # optimization
        self.learning_rate_min = train_config.get("learning_rate_min", 0.0002)
        self.clip_grad_fun = build_gradient_clipper(config=train_config)
        self.optimizer = build_optimizer(config=train_config, parameters=self.model.parameters())

        # validation & early stopping
        self.validation_freq = train_config.get("validation_freq", 10000)
        self.ckpt_best_queue = queue.Queue(maxsize=train_config.get("keep_last_ckpts", 1))
        self.ckpt_queue = queue.Queue(maxsize=1)

        self.val_on_train = config["data"].get("val_on_train", False)

        # TODO - Include Back Translation
        self.eval_metric = train_config.get("eval_metric", "bleu").lower()
        if self.eval_metric not in ['bleu', 'chrf', "dtw"]:
            raise ConfigurationError("Invalid setting for 'eval_metric', "
                                     "valid options: 'bleu', 'chrf', 'DTW'")
        self.early_stopping_metric = train_config.get("early_stopping_metric",
                                                      "eval_metric")

        if self.early_stopping_metric in ["loss", "dtw"]:
            self.minimize_metric = True
        else:
            raise ConfigurationError("Invalid setting for 'early_stopping_metric', "
                                     "valid options: 'loss', 'dtw',.")

        # learning rate scheduling
        self.scheduler, self.scheduler_step_at = build_scheduler(
            config=train_config,
            scheduler_mode="min" if self.minimize_metric else "max",
            optimizer=self.optimizer,
            hidden_size=config["model"]["encoder"]["hidden_size"])

        # data & batch handling
        self.level = "word"
        self.shuffle = train_config.get("shuffle", True)
        self.epochs = train_config["epochs"]
        self.batch_size = train_config["batch_size"]
        self.batch_type = "sentence"
        self.eval_batch_size = train_config.get("eval_batch_size", self.batch_size)
        self.eval_batch_type = train_config.get("eval_batch_type", self.batch_type)
        self.batch_multiplier = train_config.get("batch_multiplier", 1)

        # generation
        self.max_output_length = train_config.get("max_output_length", 300)

        # CPU / GPU
        self.use_cuda = train_config["use_cuda"]
        if self.use_cuda:
            self.model.cuda()
            self.loss.cuda()
            self.mulit_loss.cuda()
            self.disc.cuda()

        # initialize training statistics
        self.steps = 0
        # stop training if this flag is True by reaching learning rate minimum
        self.stop = False
        self.total_tokens = 0
        self.best_ckpt_iteration = 0
        # initial values for best scores
        self.best_ckpt_score = np.inf if self.minimize_metric else -np.inf
        # comparison function for scores
        self.is_best = lambda score: score < self.best_ckpt_score \
            if self.minimize_metric else score > self.best_ckpt_score

        ## Checkpoint restart
        # If continuing
        if model_continue:
            # Get the latest checkpoint
            ckpt = self.ckpt
            if ckpt is None:
                ckpt = get_latest_checkpoint(model_dir)
                self.logger.info("Can't find checkpoint in directory %s", ckpt)
            else:
                self.logger.info("Continuing model from %s", ckpt)
                self.init_from_checkpoint(ckpt)

        # Skip frames
        self.skip_frames = config["data"].get("skip_frames", 2)

        ## -- Data augmentation --
        # Just Counter
        self.just_count_in = config["model"].get("just_count_in", False)
        # Gaussian Noise
        self.gaussian_noise = config["model"].get("gaussian_noise", True)
        if self.gaussian_noise:
            # How much the noise is added in
            self.noise_rate = config["model"].get("noise_rate", 5)

        if self.just_count_in and (self.gaussian_noise):
            raise ConfigurationError("Can't have both just_count_in and gaussian_noise as True")

        self.future_prediction = config["model"].get("future_prediction", 0)
        if self.future_prediction != 0:
            frames_predicted = [i for i in range(self.future_prediction)]
            self.logger.info("Future prediction. Frames predicted: %s", frames_predicted)

    # Save a checkpoint
    def _save_checkpoint(self, type="every") -> None:
        # Define model path
        model_path = "{}/{}_{}.ckpt".format(self.model_dir, self.steps, type)
        # Define State
        state = {
            "steps": self.steps,
            "total_tokens": self.total_tokens,
            "best_ckpt_score": self.best_ckpt_score,
            "best_ckpt_iteration": self.best_ckpt_iteration,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if \
                self.scheduler is not None else None,
            "disc_state": self.disc.state_dict(),
            "disc_opt_state": self.disc_opt.state_dict()
        }
        torch.save(state, model_path)
        # If this is the best checkpoint
        if type == "best":
            if self.ckpt_best_queue.full():
                to_delete = self.ckpt_best_queue.get()  # delete oldest ckpt
                try:
                    os.remove(to_delete)
                except FileNotFoundError:
                    self.logger.warning("Wanted to delete old checkpoint %s but "
                                        "file does not exist.", to_delete)

            self.ckpt_best_queue.put(model_path)

            best_path = "{}/best.ckpt".format(self.model_dir)
            try:
                # create/modify symbolic link for best checkpoint
                symlink_update("{}_best.ckpt".format(self.steps), best_path)
            except OSError:
                # overwrite best.ckpt
                torch.save(state, best_path)

        # If this is just the checkpoint at every validation
        elif type == "every":
            if self.ckpt_queue.full():
                to_delete = self.ckpt_queue.get()  # delete oldest ckpt
                try:
                    os.remove(to_delete)
                except FileNotFoundError:
                    self.logger.warning("Wanted to delete old checkpoint %s but "
                                        "file does not exist.", to_delete)

            self.ckpt_queue.put(model_path)

            every_path = "{}/every.ckpt".format(self.model_dir)
            try:
                # create/modify symbolic link for best checkpoint
                symlink_update("{}_best.ckpt".format(self.steps), every_path)
            except OSError:
                # overwrite every.ckpt
                torch.save(state, every_path)

    # Initialise from a checkpoint
    def init_from_checkpoint(self, path: str) -> None:
        # Find last checkpoint
        model_checkpoint = load_checkpoint(path=path, use_cuda=self.use_cuda)

        # restore model and optimizer parameters
        self.model.load_state_dict(model_checkpoint["model_state"])
        self.optimizer.load_state_dict(model_checkpoint["optimizer_state"])
        self.disc.load_state_dict(model_checkpoint["disc_state"])
        self.disc_opt.load_state_dict(model_checkpoint["disc_opt_state"])

        if model_checkpoint["scheduler_state"] is not None and \
                self.scheduler is not None:
            # Load the scheduler state
            self.scheduler.load_state_dict(model_checkpoint["scheduler_state"])

        # restore counts
        self.steps = model_checkpoint["steps"]
        self.total_tokens = model_checkpoint["total_tokens"]
        self.best_ckpt_score = model_checkpoint["best_ckpt_score"]
        self.best_ckpt_iteration = model_checkpoint["best_ckpt_iteration"]

        # move parameters to cuda
        if self.use_cuda:
            self.model.cuda()
            self.disc.cuda()

    # Train and validate function
    def train_and_validate(self, train_data: Dataset, valid_data: Dataset, vocab: list) \
            -> None:
        train_data_iter = make_data_iter(train_data, vocab)
        val_step = 0
        if self.gaussian_noise:
            all_epoch_noise = []
        # Loop through epochs
        for epoch_no in range(self.epochs):
            train_latent_grandtrue = []
            train_latent_prediction = []
            self.logger.info("EPOCH %d", epoch_no + 1)

            if self.scheduler is not None and self.scheduler_step_at == "epoch":
                self.scheduler.step(epoch=epoch_no)
            self.model.train()
            # Reset statistics for each epoch.
            start = time.time()
            total_valid_duration = 0
            start_tokens = self.total_tokens
            count = self.batch_multiplier - 1
            epoch_loss = 0

            # If Gaussian Noise, extract STDs for each joint position
            if self.gaussian_noise:
                if len(all_epoch_noise) != 0:
                    self.pre_model.out_stds = torch.mean(
                        torch.stack(([noise.std(dim=[0]) for noise in all_epoch_noise])),
                        dim=-2)
                else:
                    self.pre_model.out_stds = None
                all_epoch_noise = []
            train_loader = DataLoader(train_data_iter, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)
            for i, (src, trg, file_path, trg_length) in enumerate(train_loader):

                # reactivate training
                self.model.train()
                #audio = torch.stack(audio)
                batch = [src, trg, file_path]
                # create a Batch object from torchtext batch
                batch = Batch(torch_batch=batch,
                              pad_index=self.pad_index,
                              model=self.model)
                batch[5] = torch.cat((batch[5][:, :, :batch[5].shape[2] // (10)], batch[5][:, :, -1:]), dim=2)
                update = count == 0
                # Train the model on a batch
                latent_output,_ = self.model.AE(trg=batch[2])
                latent_loss = self.loss(latent_output, batch[2])

                skel_out, change_out = self.model.forward(src=batch[0])#128,18

                # fgd_pred = torch.from_numpy(np.reshape(skel_out.cpu().detach().numpy()[:,:,:-1], (skel_out.size(0), skel_out.size(1), 50, 3))).cuda()
                # fgd_targ = torch.from_numpy(np.reshape(batch[2].cpu().detach().numpy()[:, :, :-1], (skel_out.size(0), skel_out.size(1), 50, 3))).cuda()
                # fgd_loss = torch.norm(fgd_targ - fgd_pred, dim=-1).mean()

                # dtw = self.dtw_loss(fgd_pred, fgd_targ)
                # change_loss = self.loss(change_out, batch[2])


                diff_pred = skel_out[:, 1:, :] - skel_out[:, :-1, :]
                diff_true = batch[5][:, 1:, :] - batch[5][:, :-1, :]
                frame_loss = self.loss(diff_pred, diff_true)

                fgd_pred = torch.from_numpy(np.reshape(skel_out.cpu().detach().numpy()[:, :, :-1],
                                                       (skel_out.size(0), skel_out.size(1), 50, 3))).cuda()
                fgd_targ = torch.from_numpy(np.reshape(batch[5].cpu().detach().numpy()[:, :, :-1],
                                                       (skel_out.size(0), skel_out.size(1), 50, 3))).cuda()
                body_pred = fgd_pred[:, :, :8, :]
                left_hand_pred = fgd_pred[:, :, 8:29, :]
                right_hand_pred = fgd_pred[:, :, 29:, :]

                body_true = fgd_targ[:, :, :8, :]
                left_hand_true = fgd_targ[:, :, 8:29, :]
                right_hand_true = fgd_targ[:, :, 29:, :]

                mulit_loss = 0.2* self.mulit_loss(body_pred, body_true) + 0.4* self.mulit_loss(left_hand_pred,
                                                                                                 left_hand_true) + 0.4* self.mulit_loss(
                    right_hand_pred, right_hand_true)
                # mulit_loss = self.mulit_loss(body_pred, body_true) + self.mulit_loss(left_hand_pred,left_hand_true) + self.mulit_loss(right_hand_pred, right_hand_true)

                batch_loss = self.loss(skel_out, batch[2]) + latent_loss + mulit_loss + frame_loss
                #batch_loss = self.loss(skel_out, batch[2]) + latent_loss

                seq_valid = Variable(Tensor(np.ones((batch[5].shape[0], 1))), requires_grad=False)
                seq_fake_gt = Variable(Tensor(np.zeros((batch[5].shape[0], 1))), requires_grad=False)

                if self.use_cuda:
                    seq_valid = seq_valid.to('cuda')
                    seq_fake_gt = seq_fake_gt.to('cuda')
                adversarial_loss = torch.nn.BCEWithLogitsLoss()
                seq_fake = self.disc(skel_out[:, :, :-1])
                disc_loss = adversarial_loss(seq_fake, seq_valid)
                batch_loss += 0.0001 * disc_loss
                if self.clip_grad_fun is not None:
                    # clip gradients (in-place)
                    self.clip_grad_fun(params=self.model.parameters())

                if update:
                    self.optimizer.zero_grad()
                    batch_loss.backward()
                    self.optimizer.step()
                    self.steps += 1

                # -----------------------------------------
                #         Train Discriminator
                # ---------------------------------------

                self.disc.zero_grad()

                # Real loss
                pred_real_seq = self.disc(batch[5][:, :, :-1])
                loss_real_seq = adversarial_loss(pred_real_seq, seq_valid)

                # Fake loss
                pred_fake_seq = self.disc(skel_out[:, :, :-1].detach())
                loss_fake_seq = adversarial_loss(pred_fake_seq, seq_fake_gt)

                # Total loss for discriminator
                if update:
                    D_loss_seq = (0.5 * (loss_real_seq + loss_fake_seq))
                    loss_D = D_loss_seq
                    loss_D.backward()
                    self.disc_opt.step()

                # increment token counter
                self.total_tokens += 1

                # batch_loss, noise, latent = self._train_batch(batch, update=update)
                # If Gaussian Noise, collect the noise
                # if self.gaussian_noise:
                #     # If future Prediction, cut down the noise size to just one frame
                #     if self.future_prediction != 0:
                #         all_epoch_noise.append(noise.reshape(-1, self.model.out_trg_size // self.future_prediction))
                #     else:
                #         all_epoch_noise.append(noise.reshape(-1, self.model.out_trg_size))

                self.tb_writer.add_scalar("train/train_batch_loss", batch_loss, self.steps)
                self.tb_writer.add_scalar("train/train_latent_loss", latent_loss, self.steps)
                self.tb_writer.add_scalar("train/train_batch_loss", batch_loss, self.steps)
                self.tb_writer.add_scalar("train/train_frame_loss", frame_loss, self.steps)
                self.tb_writer.add_scalar("train/train_mulit_loss", mulit_loss, self.steps)
                count = self.batch_multiplier if update else count
                count -= 1

                epoch_loss += batch_loss.detach().cpu().numpy()

                if self.scheduler is not None and self.scheduler_step_at == "step" and update:
                    self.scheduler.step()
                # log learning progress
                if self.steps % self.logging_freq == 0 and update:
                    elapsed = time.time() - start - total_valid_duration
                    elapsed_tokens = self.total_tokens - start_tokens
                    self.logger.info(
                        "Epoch %3d Step: %8d Batch Loss: %12.6f  Mulit_loss Loss: %12.6f "
                        "Tokens per Sec: %8.0f, Lr: %.6f",
                        epoch_no + 1, self.steps, batch_loss, mulit_loss,
                        elapsed_tokens / elapsed,
                        self.optimizer.param_groups[0]["lr"])
                    start = time.time()
                    total_valid_duration = 0
                    start_tokens = self.total_tokens

                # validate on the entire dev set
                if self.steps % self.validation_freq == 0 and update:

                    valid_start_time = time.time()

                    valid_score, valid_loss, valid_references, valid_hypotheses, \
                        valid_inputs, all_dtw_scores, valid_file_paths, talent_hypotheses, valida_length = \
                        pre_validate_on_data(
                            batch_size=self.eval_batch_size,
                            vocab=vocab,
                            data=valid_data,
                            eval_metric=self.eval_metric,
                            model=self.model,
                            max_output_length=self.max_output_length,
                            loss_function=self.loss,
                            batch_type=self.eval_batch_type,
                            type="val",
                        )

                    val_step += 1

                    # Tensorboard writer
                    self.tb_writer.add_scalar("valid/valid_loss", valid_loss, self.steps)
                    self.tb_writer.add_scalar("valid/valid_score", valid_score, self.steps)

                    if self.early_stopping_metric == "loss":
                        ckpt_score = valid_loss
                    elif self.early_stopping_metric == "dtw":
                        ckpt_score = valid_score
                    else:
                        ckpt_score = valid_score

                    new_best = False
                    self.best = False
                    if self.is_best(ckpt_score):
                        self.best = True
                        self.best_ckpt_score = ckpt_score
                        self.best_ckpt_iteration = self.steps
                        self.logger.info(
                            'Hooray! New best validation result [%s]!',
                            self.early_stopping_metric)
                        if self.ckpt_queue.maxsize > 0:
                            self.logger.info("Saving new checkpoint.")
                            new_best = True
                            self._save_checkpoint(type="best")

                        # Display these sequences, in this index order
                        # display = list(range(0, len(valid_hypotheses), int(np.ceil(len(valid_hypotheses) / 13.15))))
                        display = list(range(len(valid_hypotheses)))
                        self.produce_validation_video(
                            output_joints=valid_hypotheses,
                            inputs=valid_inputs,
                            references=valid_references,
                            model_dir=self.model_dir,
                            steps=self.steps,
                            display=display,
                            type="val_inf",
                            file_paths=valid_file_paths,
                            trg_length=valida_length
                        )
                        self.produce_validation_video(
                            output_joints=talent_hypotheses,
                            inputs=valid_inputs,
                            references=valid_references,
                            model_dir=self.model_dir,
                            steps=self.steps,
                            display=display,
                            type="talent",
                            file_paths=valid_file_paths,
                            trg_length=valida_length
                        )

                    self._save_checkpoint(type="every")

                    if self.scheduler is not None and self.scheduler_step_at == "validation":
                        self.scheduler.step(ckpt_score)
                    # append to validation report
                    self._add_report(
                        valid_score=valid_score, valid_loss=valid_loss,
                        eval_metric=self.eval_metric,
                        new_best=new_best, report_type="val", )

                    valid_duration = time.time() - valid_start_time
                    total_valid_duration += valid_duration
                    self.logger.info(
                        'Validation result at epoch %3d, step %8d: Val DTW Score: %6.2f, '
                        'loss: %8.4f,  duration: %.4fs',
                        epoch_no + 1, self.steps, valid_score,
                        valid_loss, valid_duration)

                if self.stop:
                    break

            if self.stop:
                self.logger.info(
                    'Training ended since minimum lr %f was reached.',
                    self.learning_rate_min)
                break

            self.logger.info('Epoch %3d: total training loss %.5f', epoch_no + 1,
                             epoch_loss)
        else:
            self.logger.info('Training ended after %3d epochs.', epoch_no + 1)
        self.logger.info('Best validation result at step %8d: %6.2f %s.',
                         self.best_ckpt_iteration, self.best_ckpt_score,
                         self.early_stopping_metric)

        self.tb_writer.close()  # close Tensorboard writer

    # Produce the video of Phoenix MTC joints
    def produce_validation_video(self, output_joints, inputs, references, display, model_dir, type, trg_length,
                                 steps="",
                                 file_paths=None):

        max_length = 100
        left_padding = 6
        # If not at test
        if type == "val_inf":
            dir_name = model_dir + "/videos/Step_{}/".format(steps)
            if not os.path.exists(model_dir + "/videos/"):
                os.mkdir(model_dir + "/videos/")

        elif type == "talent":
            dir_name = model_dir + "/talent_videos/Step_{}/".format(steps)
            if not os.path.exists(model_dir + "/talent_videos/"):
                os.mkdir(model_dir + "/talent_videos/")

        # If at test time
        elif type == "test":
            dir_name = model_dir + "/test_videos/"

        # Create model video folder if not exist
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        # For sequence to display
        for i in display:
            length = trg_length[i]
            right_padding = max_length - length + 6

            seq = output_joints[i][left_padding:-right_padding, :]
            ref_seq = references[i][left_padding:-right_padding, :]
            input = inputs[i]
            # Write gloss label
            gloss_label = input[0]
            if input[1] is not "</s>":
                gloss_label += "_" + input[1]
            if input[2] is not "</s>":
                gloss_label += "_" + input[2]

            # Alter the dtw timing of the produced sequence, and collect the DTW score
            timing_hyp_seq, ref_seq_count, dtw_score = alter_DTW_timing(seq, ref_seq)

            video_ext = "{}_{}.mp4".format(gloss_label, "{0:.2f}".format(float(dtw_score)).replace(".", "_"))

            if file_paths is not None:
                sequence_ID = file_paths[i]
            else:
                sequence_ID = None

            # Plot this sequences video
            if "<" not in video_ext:
                plot_video(joints=timing_hyp_seq,
                           file_path=dir_name,
                           video_name=video_ext,
                           references=ref_seq_count,
                           skip_frames=self.skip_frames,
                           sequence_ID=sequence_ID)

    # Train the batch
    def _train_batch(self, batch: Batch, update: bool = True) -> Tensor:

        # Get loss from this batch
        latent, noise = self.pre_model.get_loss_for_batch(batch=batch)
        latent_loss = self.latent_loss(latent, batch[2])
        # skel_out = self.model.decode(latent,hidden=self.model.hidden)
        skel_out, _ = self.model.forward(trg_input=batch[2], trg_mask=batch[3])
        skel_out_loss = self.loss(skel_out, batch[2])
        batch_loss = latent_loss + 0.5 * skel_out_loss
        # # normalize batch loss
        # if self.normalization == "batch":
        #     normalizer = batch.nseqs
        # elif self.normalization == "tokens":
        #     normalizer = batch.ntokens
        # else:
        #     raise NotImplementedError("Only normalize by 'batch' or 'tokens'")
        normalizer = self.batch_size
        norm_batch_loss = batch_loss / normalizer
        # division needed since loss.backward sums the gradients until updated
        norm_batch_multiply = norm_batch_loss / self.batch_multiplier

        # compute gradients
        norm_batch_multiply.requires_grad_(True)

        if self.clip_grad_fun is not None:
            # clip gradients (in-place)
            self.clip_grad_fun(params=self.model.parameters())
            self.clip_grad_fun(params=self.pre_model.parameters())

        if update:
            # make gradient step

            self.optimizer.zero_grad()
            self.latent_optimizer.zero_grad()
            # latent_loss.backward()
            # skel_out_loss.backward()
            norm_batch_multiply.backward()
            self.optimizer.step()
            self.latent_optimizer.step()
            # increment step counter
            self.steps += 1

        # increment token counter
        self.total_tokens += 1

        return batch_loss, noise, skel_out

    def _add_report(self, valid_score: float, valid_loss: float, eval_metric: str,
                    new_best: bool = False, report_type: str = "val") -> None:

        current_lr = -1
        # ignores other param groups for now
        for param_group in self.optimizer.param_groups:
            current_lr = param_group['lr']

        if current_lr < self.learning_rate_min:
            self.stop = True

        if report_type == "val":
            with open(self.valid_report_file, 'a') as opened_file:
                opened_file.write(
                    "Steps: {} Loss: {:.5f}| DTW: {:.3f}|"
                    " LR: {:.6f} {}\n".format(
                        self.steps, valid_loss, valid_score,
                        current_lr, "*" if new_best else ""))

    def _log_parameters_list(self) -> None:
        """
        Write all model parameters (name, shape) to the log.
        """
        model_parameters = filter(lambda p: p.requires_grad,
                                  self.model.parameters())
        n_params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info("Total params: %d", n_params)
        trainable_params = [n for (n, p) in self.model.named_parameters()
                            if p.requires_grad]
        self.logger.info("Trainable parameters: %s", sorted(trainable_params))
        assert trainable_params


def CVT_train(cfg_file: str, ckpt=None) -> None:
    # Load the config file
    cfg = load_config(cfg_file)
    # Set the random seed
    set_seed(seed=cfg["training"].get("random_seed", 27))
    # Load the data - Trg as (batch, # of frames, joints + 1 )
    train_data, dev_data, test_data, src_vocab, trg_vocab = load_data(cfg=cfg)
    # Build model
    model, disc = build_model(cfg, src_vocab=src_vocab)
    if ckpt is not None:
        use_cuda = cfg["training"].get("use_cuda", True)
        model_checkpoint = load_checkpoint(ckpt, use_cuda=use_cuda)
        # Build model and load parameters from the checkpoint
        model.load_state_dict(model_checkpoint["model_state"])
    # for training management, e.g. early stopping and model selection
    trainer = CVTTrainManager(model=model, disc=disc, config=cfg)

    # Store copy of original training config in model dir
    shutil.copy2(cfg_file, trainer.model_dir + "/config.yaml")
    # Log all entries of config
    log_cfg(cfg, trainer.logger)

    # Train the model
    trainer.train_and_validate(train_data=train_data, valid_data=dev_data, vocab=src_vocab)

    # Test the model with the best checkpoint
    # test(cfg_file, ckpt)


# pylint: disable-msg=logging-too-many-args
def CVT_test(cfg_file, ckpt=None) -> None:
    # Load the config file
    cfg = load_config(cfg_file)
    print('testing')

    # Load the model directory and checkpoint
    model_dir = cfg["training"]["model_dir"]
    # when checkpoint is not specified, take latest (best) from model dir
    if ckpt is None:
        ckpt = get_latest_checkpoint(model_dir, post_fix="_best")
        if ckpt is None:
            raise FileNotFoundError("No checkpoint found in directory {}."
                                    .format(model_dir))

    batch_size = cfg["training"].get("eval_batch_size", cfg["training"]["batch_size"])
    batch_type = cfg["training"].get("eval_batch_type", cfg["training"].get("batch_type", "sentence"))
    use_cuda = cfg["training"].get("use_cuda", True)
    eval_metric = cfg["training"]["eval_metric"]
    max_output_length = cfg["training"].get("max_output_length", 300)

    # load the data
    train_data, dev_data, test_data, src_vocab, trg_vocab = load_data(cfg=cfg)

    # To produce testing results
    # data_to_predict = {"test": test_data}
    # data_to_predict = make_data_iter(test_data, src_vocab)
    # To produce validation results
    # data_to_predict = {"dev": dev_data}

    # Load model state from disk
    model_checkpoint = load_checkpoint(ckpt, use_cuda=use_cuda)

    with open("dict.json", "w") as file:
        json.dump(src_vocab, file)

    # Build model and load parameters into it
    model, disc = build_model(cfg, src_vocab=src_vocab, trg_vocab=trg_vocab)
    model.load_state_dict(model_checkpoint["model_state"])
    # If cuda, set model as cuda
    if use_cuda:
        model.cuda()

    # Set up trainer to produce videos
    trainer = CVTTrainManager(model=model, disc=disc, config=cfg, test=True, ckpt=ckpt)
    # Validate for this data set
    score, loss, references, hypotheses, \
        inputs, all_dtw_scores, file_paths, talent_hypotheses, test_length, predicted_latent, reconst_latent = \
        pre_validate_on_data(
            model=model,
            data=test_data,
            batch_size=batch_size,
            max_output_length=max_output_length,
            eval_metric=eval_metric,
            loss_function=trainer.loss,
            batch_type=batch_type,
            vocab=src_vocab
        )

    predicted_latent_data = [t.tolist() for t in predicted_latent]


    # with open('predicted_latent.json', 'w') as file:
    #     json.dump(predicted_latent_data, file)
    #
    # reconst_latent_data = [t.tolist() for t in reconst_latent]
    #

    # with open('reconst_latent.json', 'w') as file:
    #     json.dump(reconst_latent_data, file)
    #
    #
    # # with open('predicted_latent.pkl', 'wb') as file:
    # #     pickle.dump(predicted_latent, file)
    # # with open('reconst_latent.pkl', 'wb') as file:
    # #     pickle.dump(reconst_latent, file)


    # Set which sequences to produce video for
    display = list(range(len(hypotheses)))

    # Produce videos for the produced hypotheses
    trainer.produce_validation_video(
        output_joints=hypotheses,
        inputs=inputs,
        references=references,
        model_dir=model_dir,
        display=display,
        type="test",
        file_paths=file_paths,
        trg_length=test_length
    )

    print("testing done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Talent Transformers")
    parser.add_argument("config", default="configs/default.yaml", type=str,
                        help="Training configuration file (yaml).")
    args = parser.parse_args()
    train(cfg_file=args.config)
