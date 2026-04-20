import gc
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from .utils import AverageMeter


class TrainDataset(Dataset):
    def __init__(self, df, tokenizer, cfg):
        self.df = df.reset_index(drop=True)
        self.texts = self.df[cfg.data.text_col].tolist()
        self.targets = self.df[cfg.data.target_col].tolist()
        self.tokenizer = tokenizer
        self.cfg = cfg

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.texts[idx]
        target = self.targets[idx]

        encoded = self.tokenizer(
            text,
            max_length=self.cfg.text.max_length,
            padding=self.cfg.text.padding,
            truncation=self.cfg.text.truncation,
            return_tensors="pt"
        )

        item = {k: v.squeeze(0) for k, v in encoded.items()}
        item["target"] = torch.tensor(target, dtype=torch.long)
        return item


class TestDataset(Dataset):
    def __init__(self, df, tokenizer, cfg):
        self.df = df.reset_index(drop=True)
        self.texts = self.df[cfg.data.text_col].tolist()
        self.tokenizer = tokenizer
        self.cfg = cfg

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.texts[idx]

        encoded = self.tokenizer(
            text,
            max_length=self.cfg.text.max_length,
            padding=self.cfg.text.padding,
            truncation=self.cfg.text.truncation,
            return_tensors="pt"
        )

        item = {k: v.squeeze(0) for k, v in encoded.items()}
        return item


def load_tokenizer(cfg, local_files_only: bool = False):
    model_path = cfg.text.local_model_path if getattr(cfg.text, "local_model_path", None) else cfg.text.model_name
    return AutoTokenizer.from_pretrained(model_path, local_files_only=local_files_only)


def create_loaders(fold, train_df, tokenizer, cfg):
    trn_df = train_df[train_df["fold"] != fold].reset_index(drop=True)
    val_df = train_df[train_df["fold"] == fold].reset_index(drop=True)

    train_dataset = TrainDataset(trn_df, tokenizer, cfg)
    valid_dataset = TrainDataset(val_df, tokenizer, cfg)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.train_bs,
        shuffle=True,
        num_workers=cfg.general.num_workers,
        pin_memory=True,
        drop_last=False
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.training.valid_bs,
        shuffle=False,
        num_workers=cfg.general.num_workers,
        pin_memory=True,
        drop_last=False
    )

    return trn_df, val_df, train_loader, valid_loader


def create_test_loader(test_df, tokenizer, cfg):
    test_dataset = TestDataset(test_df, tokenizer, cfg)

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.training.valid_bs,
        shuffle=False,
        num_workers=cfg.general.num_workers,
        pin_memory=True,
        drop_last=False
    )
    return test_loader


def compute_metrics(y_true, y_pred, n_classes: int):
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=list(range(n_classes)))
    return {
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1
    }


class BI_RADS_Classifier(nn.Module):
    def __init__(self, cfg, local_files_only: bool = False):
        super().__init__()
        self.cfg = cfg
        model_path = cfg.text.local_model_path if getattr(cfg.text, "local_model_path", None) else cfg.text.model_name

        hf_config = AutoConfig.from_pretrained(
            model_path,
            local_files_only=local_files_only
        )
        hf_config.hidden_dropout_prob = cfg.model.hidden_dropout_prob
        hf_config.attention_probs_dropout_prob = cfg.model.attention_probs_dropout_prob

        self.backbone = AutoModel.from_pretrained(
            model_path,
            config=hf_config,
            local_files_only=local_files_only
        )
        hidden_size = hf_config.hidden_size

        self.dropout = nn.Dropout(cfg.model.dropout)
        self.fc = nn.Linear(hidden_size, cfg.data.n_classes)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        if token_type_ids is not None:
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
        else:
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        cls_embedding = outputs.last_hidden_state[:, 0, :]
        logits = self.fc(self.dropout(cls_embedding))
        return logits


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction="none")

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def get_loss_fn(cfg, class_weights=None, device="cpu"):
    weights = class_weights.to(device) if class_weights is not None else None

    if cfg.loss.use_focal_loss:
        return FocalLoss(weight=weights, gamma=cfg.loss.focal_gamma)
    else:
        return nn.CrossEntropyLoss(
            weight=weights,
            label_smoothing=cfg.loss.label_smoothing
        )


def get_optimizer(model, cfg):
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    optimizer_parameters = [
        {
            "params": [p for n, p in model.backbone.named_parameters() if not any(nd in n for nd in no_decay)],
            "lr": cfg.training.encoder_lr,
            "weight_decay": cfg.training.weight_decay,
        },
        {
            "params": [p for n, p in model.backbone.named_parameters() if any(nd in n for nd in no_decay)],
            "lr": cfg.training.encoder_lr,
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in model.fc.named_parameters() if not any(nd in n for nd in no_decay)],
            "lr": cfg.training.decoder_lr,
            "weight_decay": cfg.training.weight_decay,
        },
        {
            "params": [p for n, p in model.fc.named_parameters() if any(nd in n for nd in no_decay)],
            "lr": cfg.training.decoder_lr,
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_parameters, lr=cfg.training.lr)
    return optimizer


def get_scheduler(cfg, optimizer, num_train_steps):
    num_warmup_steps = int(cfg.training.warmup_ratio * num_train_steps)

    if cfg.training.scheduler == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps,
        )
    else:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps,
        )
    return scheduler


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device, cfg):
    model.train()

    losses = AverageMeter()
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.general.use_amp)

    for step, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["target"].to(device)

        token_type_ids = batch.get("token_type_ids", None)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        with torch.cuda.amp.autocast(enabled=cfg.general.use_amp):
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            loss = criterion(logits, targets)
            loss = loss / cfg.training.gradient_accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % cfg.training.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        losses.update(loss.item() * cfg.training.gradient_accumulation_steps, input_ids.size(0))

    return losses.avg


@torch.no_grad()
def valid_one_epoch(model, loader, criterion, device, cfg):
    model.eval()

    losses = AverageMeter()
    preds = []
    probs = []
    labels = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["target"].to(device)

        token_type_ids = batch.get("token_type_ids", None)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        loss = criterion(logits, targets)

        pred = torch.argmax(logits, dim=1)
        prob = torch.softmax(logits, dim=1)

        losses.update(loss.item(), input_ids.size(0))

        preds.append(pred.detach().cpu().numpy())
        probs.append(prob.detach().cpu().numpy())
        labels.append(targets.detach().cpu().numpy())

    preds = np.concatenate(preds)
    probs = np.concatenate(probs)
    labels = np.concatenate(labels)

    metrics = compute_metrics(labels, preds, cfg.data.n_classes)

    return losses.avg, metrics, preds, probs, labels


@torch.no_grad()
def predict_test(model, loader, device, cfg):
    model.eval()

    probs = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        token_type_ids = batch.get("token_type_ids", None)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        prob = torch.softmax(logits, dim=1)
        probs.append(prob.detach().cpu().numpy())

    probs = np.concatenate(probs)
    return probs


def run_fold(fold, train_df, test_df, tokenizer, cfg, device, class_weights=None, local_files_only=False):
    print(f"\n{'='*20} FOLD {fold} {'='*20}")

    trn_df, val_df, train_loader, valid_loader = create_loaders(fold, train_df, tokenizer, cfg)
    test_loader = create_test_loader(test_df, tokenizer, cfg)

    model = BI_RADS_Classifier(cfg, local_files_only=local_files_only).to(device)
    criterion = get_loss_fn(cfg, class_weights=class_weights, device=device)
    optimizer = get_optimizer(model, cfg)

    num_train_steps = int(len(train_loader) * cfg.training.epochs / cfg.training.gradient_accumulation_steps)
    scheduler = get_scheduler(cfg, optimizer, num_train_steps)

    model_tag = os.path.basename(
        (cfg.text.local_model_path if getattr(cfg.text, "local_model_path", None) else cfg.text.model_name).rstrip("/")
    )
    best_path = os.path.join(cfg.paths.model_dir, f"{model_tag}_fold{fold}_best.pth")

    best_score = -np.inf
    best_epoch = -1
    early_stop_counter = 0
    history = []

    for epoch in range(cfg.training.epochs):
        start_time = time.time()

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            cfg=cfg
        )

        valid_loss, metrics, val_preds, val_probs, val_labels = valid_one_epoch(
            model=model,
            loader=valid_loader,
            criterion=criterion,
            device=device,
            cfg=cfg
        )

        score = metrics["macro_f1"]
        elapsed = time.time() - start_time

        history.append({
            "fold": fold,
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "macro_f1": score,
            "elapsed_sec": elapsed
        })

        print(
            f"Epoch {epoch+1}/{cfg.training.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"valid_loss={valid_loss:.4f} | "
            f"macro_f1={score:.5f} | "
            f"time={elapsed:.1f}s"
        )

        if score > best_score + cfg.early_stopping.min_delta:
            best_score = score
            best_epoch = epoch + 1
            early_stop_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_score": best_score,
                    "best_epoch": best_epoch,
                    "fold": fold,
                    "cfg_model_name": cfg.text.model_name,
                },
                best_path
            )
            print(f"Saved best model to: {best_path}")
        else:
            early_stop_counter += 1
            print(f"No improvement. Early stop counter: {early_stop_counter}/{cfg.early_stopping.patience}")

            if cfg.early_stopping.use_early_stopping and early_stop_counter >= cfg.early_stopping.patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    print(f"\nFold {fold} best macro_f1: {best_score:.5f} (epoch {best_epoch})")

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    valid_loss, metrics, val_preds, val_probs, val_labels = valid_one_epoch(
        model=model,
        loader=valid_loader,
        criterion=criterion,
        device=device,
        cfg=cfg
    )

    test_probs = predict_test(
        model=model,
        loader=test_loader,
        device=device,
        cfg=cfg
    )

    val_result = val_df[[cfg.data.id_col, cfg.data.target_col, "fold"]].copy().reset_index(drop=True)

    for c in range(cfg.data.n_classes):
        val_result[f"prob_{c}"] = val_probs[:, c]

    val_result["pred"] = np.argmax(val_probs, axis=1)

    if cfg.inference.save_probabilities:
        test_prob_path = os.path.join(cfg.paths.oof_dir, f"test_probs_fold{fold}.npy")
        np.save(test_prob_path, test_probs)
        print(f"Saved test probabilities to: {test_prob_path}")

    fold_result = {
        "fold": fold,
        "best_score": metrics["macro_f1"],
        "best_epoch": checkpoint.get("best_epoch", best_epoch),
        "val_df": val_df,
        "val_result": val_result,
        "val_labels": val_labels,
        "val_preds": val_preds,
        "val_probs": val_probs,
        "test_probs": test_probs,
        "checkpoint_path": best_path,
        "history": pd.DataFrame(history),
    }

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return fold_result
