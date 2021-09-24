from typing import Tuple, Optional
import torch
import torch.nn.functional as F 

def cosine_similarity(x1, x2):
  """Calculates cosine similarity of two tensor."""
  dist = torch.sum(torch.multiply(x1, x2), dim=-1)
  dist = dist / (torch.linalg.norm(x1, dim=-1) * torch.linalg.norm(x2, dim=-1))
  return dist

def l2_normalize(x, axis=None, epsilon=1e-12):
  square_sum = torch.sum(torch.square(x), dim=axis, keepdim=True)
  x_inv_norm = torch.rsqrt(torch.maximum(square_sum, epsilon))
  return torch.multiply(x, x_inv_norm)

def get_statistics(logits, labels):
  """Gets accuracy and entropy."""
  prob = F.softmax(logits)
  entropy = -torch.mean(torch.sum(prob * torch.log(prob + 1e-8), dim=-1))
  label_acc = torch.equal(
      torch.argmax(logits, dim=-1), torch.argmax(labels, dim=-1))
  label_acc = torch.mean(torch.Tensor(label_acc, torch.float32))
  return label_acc, entropy

def contrastive_loss(
    image_feat,
    cond_feat,
    l2_norm: bool = True,
    temperature: float = 0.1,
    sync_match: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Calculates contrastive loss."""

  if l2_norm:
    image_feat = l2_normalize(image_feat, -1)
    cond_feat = l2_normalize(cond_feat, -1)
  local_batch_size = image_feat.shape[0]
  if sync_match:
    raise NotImplementedError
  else:
    image_feat_large = image_feat
    cond_feat_large = cond_feat
    labels = F.one_hot(torch.arange(local_batch_size), local_batch_size)
    logits_img2cond = torch.matmul(image_feat,
                                 cond_feat_large.transpose()) / temperature
    logits_cond2img = torch.matmul(cond_feat,
                                 image_feat_large.transpose()) / temperature
    loss_img2cond = F.cross_entropy(
        labels=labels, logits=logits_img2cond)
    loss_cond2img = F.cross_entropy(
        labels=labels, logits=logits_cond2img)
    loss_img2cond = torch.mean(loss_img2cond)
    loss_cond2img = torch.mean(loss_cond2img)
    loss = loss_img2cond + loss_cond2img
    accuracy1, entropy1 = get_statistics(logits_img2cond, labels)
    accuracy2, entropy2 = get_statistics(logits_cond2img, labels)
    accuracy = 0.5 * (accuracy1 + accuracy2)
    entropy = 0.5 * (entropy1 + entropy2)
    return loss, accuracy, entropy

def attention(region_feat, word_feat, gamma, mask=None):
  """Calculates region attention.
  Args:
    region_feat: Regions features of shape (batch_size, region_num, feat_dim)
    word_feat: Word features of shape (batch_size, word_dim, feat_dim)
    gamma: Gamma used in the softmax shappenning function.
    mask: For masking padding word.
  Returns:
    region_context: For each word, its aggregated region context.
    alpha: The attention weights.
  """
  region_feat = l2_normalize(region_feat, -1)
  word_feat = l2_normalize(word_feat, -1)
  # batch_size * region_num * word_dim
  attn_matrix = torch.matmul(region_feat, word_feat.transpose((0, 2, 1)))
  attn_matrix = attn_matrix * gamma
  if mask is not None:
    attn_matrix = attn_matrix + mask * (-1e9)
  alpha = F.softmax(attn_matrix, dim=-2)
  region_context = torch.matmul(alpha.transpose((0, 2, 1)), region_feat)
  return region_context

def word_loss(image_feat, word_feat, max_len, gamma1=5, gamma2=5, gamma3=50):
  """Computes the word-level contrastive loss.
  Args:
    image_feat: Image features has shape (batch_size, region_num, feat_dim)
    word_feat: Word features has shape (batch_size, word_dim, feat_dim)
    max_len: The number of total words for each sentence. (batch_size,)
    gamma1: Gamma1 used in attnGAN paper.
    gamma2: Gamma2 used in attnGAN paper.
    gamma3: Gamma3 used in attnGAN paper.
  Returns:
    matching_loss: The word level matching loss.
    accuracy: The matching accuracy.
    entropy: The prediction entropy.
  """
  batch_size, region_num, _ = image_feat.shape
  total_len = word_feat.shape[1]

  def my_func(max_len_i, word_feat_i):
    word_feat_i = word_feat_i[None, :]
    word_feat_i = torch.tile(word_feat_i, (batch_size, 1, 1))
    max_len_i = torch.tile(max_len_i, region_num)
    mask = torch.arange(
        total_len, dtype=torch.float32)[None, :] >= max_len_i[:, None]
    mask = torch.Tensor(mask, dtype=torch.float32)
    mask = mask[None, :]
    mask = torch.tile(mask, (batch_size, 1, 1))
    mask_2 = mask[:, 0, :]
    # (batch_size, word_dim, feat_dim)
    region_context = attention(image_feat, word_feat_i, gamma1, mask)
    row_sim = cosine_similarity(word_feat_i, region_context)
    row_sim = row_sim * gamma2  # (batch_size, word_dim)
    row_sim = row_sim + mask_2 * (-1e9)
    row_sim = torch.logsumexp(row_sim, dim=-1, keepdim=True)
    row_sim = row_sim / gamma2
    return row_sim

  similarities = torch.vmap(my_func)(max_len, word_feat)
  similarities = similarities * gamma3
  similarities = torch.squeeze(similarities)
  similarities_transpose = similarities  # To be consistent with tf
  similarities = similarities_transpose.transpose()

  labels = F.one_hot(torch.arange(batch_size), batch_size)
  loss_0 = F.cross_entropy(
      target=labels, input=similarities)
  loss_1 = F.cross_entropy(
      target=labels, input=similarities_transpose)
  loss_0 = torch.mean(loss_0)
  loss_1 = torch.mean(loss_1)
  matching_loss = loss_0 + loss_1
  accuracy1, entropy1 = get_statistics(
      similarities, labels
  )  # different from tf, calculates accuracy and entropy from two sides
  accuracy2, entropy2 = get_statistics(
      similarities_transpose, labels
  )  # different from tf, calculates accuracy and entropy from two sides
  accuracy = 0.5 * (accuracy1 + accuracy2)
  entropy = 0.5 * (entropy1 + entropy2)
  return matching_loss, accuracy, entropy