import matplotlib.pyplot as plt

def visualize_samples(dataset, num=3):
    fig, axes = plt.subplots(num, 3, figsize=(12, 4 * num))
    for i in range(num):
        img, mask, nodes, edges = dataset[i]
        
        # Image
        axes[i, 0].imshow(img.permute(1, 2, 0))
        axes[i, 0].set_title("Input RGB")
        
        # Mask
        axes[i, 1].imshow(mask, cmap='jet')
        axes[i, 1].set_title("Segmentation")
        
        # Scene Graph Overlay
        axes[i, 2].imshow(img.permute(1, 2, 0))
        for edge in edges:
            p1 = nodes[edge[0], 1:]
            p2 = nodes[edge[1], 1:]
            axes[i, 2].plot([p1[0], p2[0]], [p1[1], p2[1]], 'w-', alpha=0.8, lw=2)
        
        for node in nodes:
            axes[i, 2].scatter(node[1], node[2], c='red', s=60)
            
        axes[i, 2].set_title("Scene Graph")
    plt.tight_layout()
    plt.show()