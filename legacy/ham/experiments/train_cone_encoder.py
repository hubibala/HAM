import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# Force JAX to use CPU only
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from jax import random, jit, value_and_grad, vmap
import optax
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from ham.models import RaceCarEncoder
from ham.sim.racecar import EgoRaceCarWrapper


# --- 1. Data Collection (Unsupervised / Self-Supervised) ---
def collect_diverse_data(num_steps=3000):
    print("Collecting Diverse Ego-Centric Data...")
    env = EgoRaceCarWrapper(render_mode="rgb_array")
    obs, _ = env.reset()
    data = []

    # We want diversity: Road, Grass, Curbs, different angles
    for i in tqdm(range(num_steps)):
        action = env.action_space.sample()
        # Bias towards movement but allow chaos
        action[1] = 0.4
        action[0] = np.sin(i / 15.0)  # Wiggle to see angles

        obs, _, term, trunc, info = env.step(action)

        # Save Observation and Surface Label (for visualization only)
        label = 0  # Road
        if info.get("surface") == "ice":
            label = 1
        if info.get("surface") == "grass":
            label = 2

        data.append({"img": obs, "label": label})

        if term or trunc:
            obs, _ = env.reset()
    env.close()
    return data


# --- 2. Data Augmentation (The Key to SimCLR) ---
def augment(image, key):
    # Image: (64, 64, 9)
    # We apply random noise and slight color jitter to create "Views"
    noise = random.normal(key, image.shape) * 0.05
    return jnp.clip(image + noise, 0.0, 1.0)


class SimCLRDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]["img"], self.data[idx]["label"]


# --- 3. Trainer (SimCLR) ---
class SimCLRTrainer:
    def __init__(self, latent_dim=3, lr=1e-3, temperature=0.1):
        self.model = RaceCarEncoder(latent_dim=latent_dim)
        self.tx = optax.adam(lr)
        self.temp = temperature

        model = self.model
        tx = self.tx
        temp = self.temp

        def train_step_logic(params, opt_state, key, batch_imgs):
            # Generate two views for each image
            k1, k2 = random.split(key)
            view1 = vmap(augment)(batch_imgs, random.split(k1, batch_imgs.shape[0]))
            view2 = vmap(augment)(batch_imgs, random.split(k2, batch_imgs.shape[0]))

            def loss_fn(p):
                z1 = model.apply(p, view1)  # (B, D)
                z2 = model.apply(p, view2)  # (B, D)

                # SimCLR Loss (NT-Xent)
                # Concatenate: [z1, z2] -> (2B, D)
                z = jnp.concatenate([z1, z2], axis=0)

                # Similarity Matrix
                sim = jnp.dot(z, z.T) / temp

                # Mask out self-similarity
                mask = jnp.eye(2 * batch_imgs.shape[0])
                sim = sim - mask * 1e9

                # Positive pairs: (i, i+B) and (i+B, i)
                target_idx = jnp.concatenate(
                    [
                        jnp.arange(batch_imgs.shape[0]) + batch_imgs.shape[0],
                        jnp.arange(batch_imgs.shape[0]),
                    ]
                )

                loss = optax.softmax_cross_entropy(
                    sim, jax.nn.one_hot(target_idx, 2 * batch_imgs.shape[0])
                )
                return jnp.mean(loss)

            l, g = value_and_grad(loss_fn)(params)
            updates, new_opt = tx.update(g, opt_state)
            return optax.apply_updates(params, updates), new_opt, l

        self.step = jit(train_step_logic)

    def init(self, key, sample):
        params = self.model.init(key, sample)
        opt = self.tx.init(params)
        return params, opt


# --- 4. Visualization ---
def visualize_clusters(params, model, dataloader):
    print("Visualizing Content Clusters...")
    embeddings, labels = [], []

    for img, label in dataloader:
        z = model.apply(params, jnp.array(img.numpy()))
        embeddings.append(np.array(z))
        labels.append(label.numpy())

    z = np.concatenate(embeddings, axis=0)
    l = np.concatenate(labels, axis=0)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Wireframe Sphere
    u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z_sph = np.cos(v)
    ax.plot_wireframe(x, y, z_sph, color="gray", alpha=0.1)

    # Scatter by Class
    # 0=Road (Black), 1=Ice (Cyan), 2=Grass (Green)
    colors = ["black", "cyan", "green"]
    names = ["Road", "Ice", "Grass"]

    for i in range(3):
        mask = l == i
        if np.sum(mask) > 0:
            ax.scatter(
                z[mask, 0], z[mask, 1], z[mask, 2], c=colors[i], label=names[i], alpha=0.5, s=15
            )

    ax.set_title("SimCLR Content Manifold")
    ax.legend()
    plt.savefig("simclr_manifold.png")
    print("Saved simclr_manifold.png")


if __name__ == "__main__":
    key = random.PRNGKey(42)

    # 1. Collect
    data = collect_diverse_data(3000)
    ds = SimCLRDataset(data)
    dl = DataLoader(ds, batch_size=128, shuffle=True, drop_last=True)
    vis_dl = DataLoader(ds, batch_size=200, shuffle=True)

    # 2. Train
    trainer = SimCLRTrainer()
    sample = jnp.array(data[0]["img"][None, ...])
    params, opt = trainer.init(key, sample)

    print("Training SimCLR...")
    for epoch in range(10):
        losses = []
        for img, _ in dl:
            key, subk = random.split(key)
            img = jnp.array(img.numpy())
            params, opt, l = trainer.step(params, opt, subk, img)
            losses.append(l)
        print(f"Epoch {epoch}: Loss {np.mean(losses):.4f}")

    # 3. Save
    visualize_clusters(params, trainer.model, vis_dl)

    import pickle

    with open("encoder_params.pkl", "wb") as f:
        pickle.dump(params, f)
    print("Saved encoder_params.pkl")
