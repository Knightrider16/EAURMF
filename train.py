import os
import argparse
import torch
import pickle as pkl
import numpy as np
import torch.nn as nn
import torch.optim as optim
import logging
import time


from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm


from losses import multimodalloss_js, multimodalloss, multimodal_wasserstein, multimodalloss_mmd, con_loss, KL_regular
from src.data.helpers import get_data_loaders
from src.models import get_model
from src.utils.utils import set_seed, autodetect_device
from src.utils.config import Config
from src.utils.checkpoint import CheckpointManager

device = autodetect_device()


parser = argparse.ArgumentParser(description="Train Models")

parser.add_argument("--batch_sz", type=int, default=8)
parser.add_argument("--bert_model", type=str, default="./prebert")
parser.add_argument("--data_path", type=str, default="./datasets/MVSA_Single/")
parser.add_argument("--drop_img_percent", type=float, default=0.0)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--embed_sz", type=int, default=300)
parser.add_argument("--freeze_img", type=int, default=3)
parser.add_argument("--freeze_txt", type=int, default=5)
parser.add_argument("--glove_path", type=str, default="./datasets/glove_embeds/glove.840B.300d.txt")
parser.add_argument("--gradient_accumulation_steps", type=int, default=40)
parser.add_argument("--hidden", nargs="*", type=int, default=[])
parser.add_argument("--hidden_sz", type=int, default=768)
parser.add_argument("--img_embed_pool_type", type=str, default="avg", choices=["max", "avg"])
parser.add_argument("--img_hidden_sz", type=int, default=2048)
parser.add_argument("--include_bn", type=int, default=True)
parser.add_argument("--lr", type=float, default=5e-05)
parser.add_argument("--lr_factor", type=float, default=0.5)
parser.add_argument("--lr_patience", type=int, default=2)
parser.add_argument("--max_epochs", type=int, default=15)
parser.add_argument("--max_seq_len", type=int, default=512)
parser.add_argument("--model", type=str, default="latefusion", choices=["bow", "img", "bert", "concatbow", "concatbert", "mmbt","latefusion"])
parser.add_argument("--n_workers", type=int, default=4)
parser.add_argument("--name", type=str, default="URMF")
parser.add_argument("--num_image_embeds", type=int, default=3)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--savedir", type=str, default="./saved/MVSA_Single")
parser.add_argument("--seed", type=int, default=1699)
parser.add_argument("--task", type=str, default="MVSA_Single", choices=["MVSA_Single"])
parser.add_argument("--task_type", type=str, default="classification", choices=["multilabel", "classification"])
parser.add_argument("--warmup", type=float, default=0.1)
parser.add_argument("--weight_classes", type=int, default=1)
parser.add_argument("--df", type=bool, default=True)
parser.add_argument("--noise", type=float, default=0.0)
parser.add_argument("--log_marker", type=str, default='')
parser.add_argument("--modulation_starts", type=int, default=5)
parser.add_argument("--modulation_ends", type=int, default=100)
parser.add_argument("--zeta", type=float, default=0.01)
parser.add_argument("--loss", type=str, default='default')
parser.add_argument("--debug", action="store_true", help="Run in debug mode")


args, remaining_args = parser.parse_known_args()
assert remaining_args == [], remaining_args

config = Config()

root_logger = logging.getLogger()
if not root_logger.handlers:
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler()],
    )
logger = logging.getLogger(__name__)


losses_map = {
    'mmd': multimodalloss_mmd,
    'default': multimodalloss,
    'wasserstein': multimodal_wasserstein,
    'js': multimodalloss_js
}

mmloss = losses_map[args.loss]

def get_criterion(args):

    criterion = nn.CrossEntropyLoss()

    return criterion


def get_optimizer(model, args):

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    return optimizer


def get_scheduler(optimizer, args):
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", patience=args.lr_patience, factor=args.lr_factor
    )



def model_eval(data, model, criterion=None):
    with torch.no_grad():
        losses, preds, tgts = [], [], []
        for batch in data:
            loss, out, tgt, cog_un = model_forward(model, batch, mode='eval')
            losses.append(loss.item())


            pred = torch.nn.functional.softmax(out, dim=1).argmax(dim=1).cpu().detach().numpy()

            preds.append(pred)
            tgt = tgt.cpu().detach().numpy()
            tgts.append(tgt)

    metrics = {}

    tgts = [l for sl in tgts for l in sl]
    preds = [l for sl in preds for l in sl]
    metrics["acc"] = accuracy_score(tgts, preds)
    metrics["f1"] = f1_score(tgts, preds, average="weighted")
    metrics["loss"] = np.mean(losses)


    return metrics


def model_forward(model, batch, mode=None):
    txt, segment, mask, img, tgt, idx = batch


    txt, img = txt.to(device), img.to(device)
    mask, segment = mask.to(device), segment.to(device)
    out = model(txt, mask, segment, img)
    txt_img_logits, txt_logits, img_logits, txt_mu, txt_logvar, img_mu, img_logvar, mu, logvar, z, cog_un = out


    tgt = tgt.to(device)

    conloss = con_loss(txt_mu,torch.exp(txt_logvar),img_mu,torch.exp(img_logvar))
    loss = mmloss(
        txt_img_logits, 
        txt_logits, 
        tgt, 
        img_logits, 
        txt_mu, txt_logvar,
        img_mu, img_logvar,
        mu, logvar,
        z
    )
    loss += 1e-5 * KL_regular(txt_mu, txt_logvar, img_mu, img_logvar) + conloss * 1e-3

    return loss, txt_img_logits, tgt, cog_un

def train(args):
    
    config.description = str(input("Enter training description (Press enter to leave blank description): "))
    
    checkpointManager = CheckpointManager(config)
    set_seed(args.seed)
    args.savedir = os.path.join(args.savedir, args.name)
    os.makedirs(args.savedir, exist_ok=True)

    train_loader, val_loader, test_loader = get_data_loaders(args)

    model = get_model(args)

    criterion = get_criterion(args)
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)



    argsDict = args.__dict__
    for eachArg, value in argsDict.items():
        logger.info('{}:{}'.format(eachArg, value))
    logger.info("==============================================")
    logger.info(f"Using {args.loss} multimodal loss function {mmloss}")
    logger.info("==============================================")
    

    model.to(device)


    torch.save(args, os.path.join(args.savedir, "args.pt"))

    start_epoch, global_step, n_no_improve, best_metric = 0, 0, 0, -np.inf



    logger.info("Training..")
    

    for i_epoch in range(start_epoch, args.max_epochs):
        logger.info(f"epoch {i_epoch + 1}/{args.max_epochs}")
        train_losses = []
        model.train()
        # optimizer.zero_grad()
        preds, tgts = [], []
        
        for batch in tqdm(train_loader, total=len(train_loader)):
        # for batch in train_loader:
            loss, out, tgt, cog_uncertainty_dict = model_forward(model, batch, mode='train')
            train_losses.append(loss.item())



            loss.backward()

            if args.modulation_starts <= i_epoch <= args.modulation_ends: # bug fixed
                coeff_l = args.zeta * cog_uncertainty_dict['l'].mean()
                coeff_v = args.zeta * cog_uncertainty_dict['v'].mean()
                for name, parms in model.named_parameters():
                    if parms.grad == None:
                        continue
                    if any( _ in name for _ in ["txtclf"]):
                        parms.grad = parms.grad * (1+coeff_v)
                    if any( _ in name for _ in ["imgclf"]):
                        parms.grad = parms.grad * (1+coeff_l)
            
            
            global_step += 1
            if global_step % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            pred = torch.nn.functional.softmax(out, dim=1).argmax(dim=1).cpu().detach().numpy()

            preds.append(pred)
            tgt = tgt.cpu().detach().numpy()
            tgts.append(tgt)

        tgts = [l for sl in tgts for l in sl]
        preds = [l for sl in preds for l in sl]
        train_acc = accuracy_score(tgts, preds)
        train_f1 = f1_score(tgts, preds, average="weighted")

        ## validation
        model.eval()
        val_metrics = model_eval(val_loader, model)
        
        scheduler.step((val_metrics['acc']))

        # Testing
        model.eval()
        test_metrics = model_eval(test_loader, model)
        
        metrics = {
            'train': {
                'loss': np.mean(train_losses),
                'acc': train_acc,
                'f1': train_f1
            },
            'test': test_metrics,
            'val': val_metrics
        }

        
        logger.info(f"Train Loss: {np.mean(train_losses):.5f} | Acc: {train_acc:.5f} | f1: {train_f1}")
        logger.info(f"Val Loss: {val_metrics['loss']:.5f} | Acc: {val_metrics['acc']:.5f} | f1: {val_metrics['f1']}")
        logger.info(f"Test Loss: {test_metrics['loss']:.5f} | Acc: {test_metrics['acc']:.5f} | f1: {test_metrics['f1']}")
        

        checkpoint_data = {
            'epoch': i_epoch + 1,
            'model_state_dict': model.state_dict() if i_epoch + 1 == args.max_epochs else None,
            'optimizer_state_dict':  optimizer.state_dict() if i_epoch + 1 == args.max_epochs else None,
            'scheduler_state_dict': scheduler.state_dict() if i_epoch + 1 == args.max_epochs else None,
            'metrics': metrics
        }
        
        checkpointManager.save(f'epoch_{i_epoch + 1}.pt', checkpoint_data)

        model.train()

    model.save('saved')




def cli_main():

    
    train(args)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    cli_main()
