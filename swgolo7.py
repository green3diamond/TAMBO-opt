

"""## Optimization

This is part where we start moving our arrays according to Utility Function.

You can check the utilities that we use above and see that it is > 0

So we will use -U as a "loss" function and minimize it, so that we can maximize our value
"""

#First save the weights or load them so that we don't have to train the network every time we start the kernel
#Remove the commend operator "#" if you want to save

#torch.save(model.state_dict(), './NN_files/model_weights.pth')

model = Reconstruction()

path = Path("./NN_Files/checkpoint.pth")

if path.exists():
    checkpoint = torch.load('./NN_Files/checkpoint.pth')

    model.load_state_dict(checkpoint['model_state_dict'])
    print("Updated Weights are loaded")

else:
    model.load_state_dict(torch.load('./NN_files/model_weights.pth'))
    print("Initial Weights are loaded")

def barycentric_coords(P, A, B, C):
    """
    Compute barycentric coordinates for each point P with respect to triangle ABC.
    P: Tensor of shape (N, 2)
    A, B, C: Tensors of shape (2,)
    """
    v0 = C - A
    v1 = B - A
    v2 = P - A

    d00 = v0 @ v0
    d01 = v0 @ v1
    d11 = v1 @ v1
    d20 = torch.sum(v2 * v0, dim=1)
    d21 = torch.sum(v2 * v1, dim=1)

    denom = d00 * d11 - d01 * d01 + 1e-8
    u = (d11 * d20 - d01 * d21) / denom
    v = (d00 * d21 - d01 * d20) / denom
    return u, v

def project_to_triangle(x, y):
    """
    Projects each (x[i], y[i]) point inside the triangle defined by:
    A = (-3200, 2000), B = (1800, 2000), C = (1800, -3600)
    x, y: tensors of shape (N,)
    Returns: projected x and y, tensors of shape (N,)
    """
    A = torch.tensor([-3800.0, 1500.0], device=x.device)
    B = torch.tensor([1200.0, 1500.0], device=x.device)
    C = torch.tensor([1200.0, -4100.0], device=x.device)

    P = torch.stack([x, y], dim=1)  # (N, 2)

    # Compute barycentric coordinates
    u, v = barycentric_coords(P, A, B, C)

    # Determine which points are inside the triangle
    inside = (u >= 0) & (v >= 0) & (u + v <= 1)

    # Clip to triangle: u, v ∈ [0, 1], u + v ≤ 1
    u_clipped = torch.clamp(u, 0.0, 1.0)
    v_clipped = torch.clamp(v, 0.0, 1.0)
    uv_sum = u_clipped + v_clipped
    over = uv_sum > 1.0
    u_clipped[over] = u_clipped[over] / uv_sum[over]
    v_clipped[over] = v_clipped[over] / uv_sum[over]

    v0 = C - A
    v1 = B - A
    P_proj = A + u_clipped.unsqueeze(1) * v0 + v_clipped.unsqueeze(1) * v1

    # If already inside, keep P. Otherwise, use projection.
    final_P = torch.where(inside.unsqueeze(1), P, P_proj)

    return final_P[:, 0], final_P[:, 1]  # x_proj, y_proj

e = 0

for i in range(1000):
    p_layout = Path(f"./Python_Layouts/Layout_{i + 1}.txt")

    if p_layout.exists():
        data = np.loadtxt(p_layout)

        x = torch.tensor(data[:, 0], dtype = torch.float32)
        y = torch.tensor(data[:, 1], dtype = torch.float32)
        e = i + 1

if e > 0:
    print(f"Updated Layout {e} is initialized")

else:
    print("First Layout is initialized")

class LearnableXY(torch.nn.Module):
    def __init__(self, x_init, y_init):
        super().__init__()
        self.x = torch.nn.Parameter(x_init)
        self.y = torch.nn.Parameter(y_init)

    def forward(self):
        return self.x, self.y

xy_module = LearnableXY(x, y)

def push_apart(module, min_dist = 2 * TankRadius):
    x, y = module()  # Correctly calls forward()
    coords = torch.stack([x, y], dim=1)  # shape (N, 2)

    with torch.no_grad():
        for i in range(coords.shape[0]):
            diffs = coords[i] - coords
            dists = torch.norm(diffs, dim=1)
            mask = (dists < min_dist) & (dists > 0)

            for j in torch.where(mask)[0]:
                direction = diffs[j] / dists[j]
                displacement = 0.5 * (min_dist - dists[j]) * direction
                coords[i] += displacement
                coords[j] -= displacement

        # Update learnable parameters in-place
        module.x.data.copy_(coords[:, 0])
        module.y.data.copy_(coords[:, 1])

# Commented out IPython magic to ensure Python compatibility.
# %%time
# 
# #Now we do the optimization step
# SWGOopt = True
# optimizer = torch.optim.SGD(xy_module.parameters(), lr = 5, momentum = .9)
# 
# if path.exists():
#     optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
#     num_epoch = checkpoint.get("epoch") + 1
#     loss = checkpoint.get("loss")
# 
#     print(f"Optimizer is initialized from the last epoch {num_epoch}")
# 
# else:
#     num_epoch = 0
#     print("Optimizer is initialized")
# 
# max_grad = 10.
# 
# Nbatch = 500
# 
# if num_epoch < 20:
#     Nfinetune = 2500
# 
# elif num_epoch >= 20:
#     Nfinetune = 5000
# 
# U_vals = []
# U_pr_vals = []
# U_e_vals = []
# U_th_vals = []
# 
# for epoch in range(num_epoch, num_epoch + 100):
#     x1, y1 = xy_module()
# 
#     #Batch Generation:
#     N_list = []
#     T_list = []
#     labels_batch = torch.zeros((Nbatch, 5))
# 
#     for i in range(Nbatch):
#         N, T, x0, y0, E, theta, phi = GenerateShowers(x1, y1)
# 
#         N_list.append(N)
#         T_list.append(T)
#         labels_batch[i] = torch.tensor([x0, y0, E, theta, phi], dtype = torch.float32)
# 
#     x = x1.unsqueeze(0).repeat(Nbatch, 1)
#     y = y1.unsqueeze(0).repeat(Nbatch, 1)
# 
#     events_n_batch = torch.stack(N_list).squeeze(-1)  # shape: (Nbatch, Nunits)
#     events_t_batch = torch.stack(T_list).squeeze(-1)
# 
#     events_batch = torch.stack((x, y, events_n_batch, events_t_batch), dim = 2)
# 
#     model.eval()
# 
#     preds_batch = model(events_batch.view(Nbatch, -1))
# 
#     #Denormalize the predictions
#     preds_x = preds_batch[:, 0] * 5000
#     preds_y = preds_batch[:, 1] * 5000
#     preds_e, preds_th, preds_phi = DenormalizeLabels(preds_batch[:, 2], preds_batch[:, 3],
#                                                                                 preds_batch[:, 4])
# 
#     #Compute the reconstructability score for each event:
#     r_score = reconstructability(events_batch[:, :, 2])
#     density = Nbatch / ((labels_batch[:, 0].max() - labels_batch[:, 0].min()) *
#                         (labels_batch[:, 1].max() - labels_batch[:, 1].min()))
# 
#     #Compute Utility:
#     U = 1e-2 * U_TH(preds_th, labels_batch[:, 3], r_score) + U_E(preds_e, labels_batch[:, 2], r_score) + U_PR(r_score) / torch.sqrt(density)
# 
#     #Save the utility values to plot
#     U_vals.append(U.item())
#     U_pr_vals.append((U_PR(r_score) / torch.sqrt(density)).item())
#     U_e_vals.append(U_E(preds_e, labels_batch[:, 2], r_score).item())
#     U_th_vals.append(1e-2 * U_TH(preds_th, labels_batch[:, 3], r_score).item())
# 
#     #sym_loss = symmetry_loss(x1, y1, n_symmetry = 3)
#     print(f"Utility: {U:.2f}")
# 
#     #Use utility as - Loss, we have a penalty term to make configuration symmetric
#     Loss = - U #+ lambda_sym * sym_loss
# 
#     Loss.backward()
# 
#     torch.nn.utils.clip_grad_norm_(xy_module.parameters(), max_norm = max_grad)
# 
#     optimizer.step()
# 
#     with torch.no_grad():
#         push_apart(xy_module)
# 
#     #This part is done to keep the detectors inside the site
#     """
#     with torch.no_grad():
#         x_proj, y_proj = project_to_triangle(x1.view(-1), y1.view(-1))
#         x1.copy_(x_proj)
#         y1.copy_(y_proj)
#     """
# 
#     optimizer.zero_grad()
# 
#     #Now we need to generate new events that with the new layout so that we can fine tune our NN when it's necessary
#     if (epoch + 1) % 5 == 0:
#         print(f"Fine Tune at epoch {epoch + 1}")
# 
#         finetune_events = torch.zeros((Nfinetune, Nunits, 4))
#         finetune_trues = torch.zeros((Nfinetune, 5))
# 
#         with torch.no_grad():
#             x2, y2 = xy_module()
#             x = x2.detach()
#             y = y2.detach()
# 
#             #Generate events to fine tune the network
#             for i in range(Nfinetune):
#                 N, T, x0, y0, E, theta, phi = GenerateShowers(x, y)
# 
#                 x0 /= 5000
#                 y0 /= 5000
#                 E, theta, phi = NormalizeLabels(E, theta, phi)
# 
#                 finetune_events[i] = torch.tensor(np.column_stack((x, y, N, T)))
#                 finetune_trues[i] = torch.tensor([x0, y0, E, theta, phi], dtype = torch.float32)
# 
#         ReconstructionNN = model
#         ReconstructionNN.train()
#         criterion = nn.MSELoss()
#         optimizerNN = torch.optim.Adam(ReconstructionNN.parameters(), lr = 5e-5)
# 
#         dataset = TensorDataset(finetune_events, finetune_trues)
#         dataloader = DataLoader(dataset, batch_size = 32, shuffle = True, drop_last = True, num_workers = 4)
# 
#         for j in range(5):
#             for ft_batch, ft_trues in dataloader:
#                 batch_size = ft_batch.size(0)
# 
#                 train_x = ft_batch.view(batch_size, -1)
# 
#                 train_y = ft_trues.view(batch_size, 5)
# 
#                 #Train the network
#                 outputs = ReconstructionNN(train_x)
# 
#                 lossT = criterion(outputs, train_y)
# 
#                 lossT.backward()
#                 optimizerNN.step()
# 
#                 optimizerNN.zero_grad()
# 
#     #I will save the layouts and weights here so that I can stop whenever and continue later
#     torch.save({"epoch": epoch, "loss": Loss, "model_state_dict": model.state_dict(),
#                 "optimizer_state_dict": optimizer.state_dict()}, "./NN_Files/checkpoint.pth")
# 
#     np.savetxt(f"./Python_Layouts/Layout_{epoch + 1}.txt", np.column_stack((x1.detach().numpy(), y1.detach().numpy())))

a, b = Layouts()

plt.figure(figsize = [6, 6])
plt.scatter(x.detach(), y.detach(), color = "red", alpha = .8, label = "Final Layout")
plt.scatter(a, b, color = "blue", alpha = .3, label = "Initial Layout")
#plt.plot([-3800, 1200, 1200, -3800], [1500, 1500, -4100, 1500], "k--", label = "SWGO Site")

#Add Shower Simulation Area
#ax = plt.gca()
#rect = patches.Rectangle((-3000, -3000), 6000, 6000, linewidth=1.5, edgecolor='green', facecolor='green',
                         #alpha=0.1, label='Shower Simulation Area')
#ax.add_patch(rect)

plt.grid()
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title("Position of the detectors")
plt.legend()
plt.xlim((-3000, 3000))
plt.ylim((-3000, 3000))
plt.show()

utility_path = Path("./Python_Layouts/Utilities.txt")

if utility_path.exists():
    data = np.loadtxt(utility_path)
    u_t = data
    #u_p = data[:, 1]
    #u_e = data[:, 2]
    #u_a = data[:, 3]

    u_t = np.append(u_t, np.array(U_vals).ravel())
    #u_p = np.append(u_p, np.array(U_pr_vals).ravel())
    #u_e = np.append(u_e, np.array(U_e_vals).ravel())
    #u_a = np.append(u_a, np.array(U_th_vals).ravel())

    #data = np.column_stack((u_t, u_p, u_e, u_a))

    np.savetxt(utility_path, u_t)

else:
    u_t = U_vals
    #u_p = U_pr_vals
    #u_e = U_e_vals
    #u_a = U_th_vals

    #data = np.column_stack((u_t, u_p, u_e, u_a))
    np.savetxt(utility_path, u_t)

mean_ut = [np.mean(u_t[i - 4: i]) for i in range(4, len(u_t))]

plt.plot(u_t, color = "black", linestyle = "none", marker = "o", linewidth = .5, label = "Total Utility")
plt.plot(np.arange(4, len(u_t)), mean_ut, linestyle = "-", linewidth = 2, label = "Mean Utility")
#plt.plot(u_p, color = "orange", linestyle = "-.", label = "Reconstructability Utility")
#plt.plot(u_e, color = "green", linestyle = "-.", label = "Energy Utility")
#plt.plot(u_a, color = "purple", linestyle = "-.", label = "Angle Utility")
plt.legend()
plt.grid()
plt.xlabel("Epoch")
plt.ylabel("Utility Score")
plt.ylim((0, np.max(u_t) + 1000))
plt.title("Utility Values per Optimization Step")
plt.show()

