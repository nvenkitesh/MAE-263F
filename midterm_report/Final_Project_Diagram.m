%% draw_model_figure.m
% Opens a MATLAB figure showing the planar DER solar array model with
% nodal DOFs, elastic/damping force vectors, boundary conditions, and
% energy values drawn directly on the figure.

clear; clc; close all;

%% 1. Physical & Discretization Parameters
nPanels    = 3;
nHinges    = nPanels - 1;
L_panel    = 1.0;          % Rigid panel length [m]
L_hinge    = 0.12;         % Compliant hinge length [m]
EA_panel   = 1.0e6;        % Panel axial stiffness [N]
EA_hinge   = 1.0e5;        % Hinge axial stiffness [N]
EI_panel   = 1.0e4;        % Panel bending stiffness [N*m^2]
EI_hinge   = 5.0;          % Hinge bending stiffness [N*m^2]
c_damp     = 2.0;          % Viscous damping coefficient [N*s/m]
rootBC     = 'clamped';    % 'clamped' or 'pinned'

phi_init   = [0.45, -0.55, 0.50, -0.45]; % Initial folded turning angles [rad]
phi_target = zeros(1, 2*nPanels - 2);    % Deployed target turning angles [rad]

%% 2. Geometry, DOFs & Energies
segLengths = []; segEA = []; segEI = []; segTypes = {};
for p = 1:nPanels
    segLengths(end+1) = L_panel; segEA(end+1) = EA_panel; segEI(end+1) = EI_panel; segTypes{end+1} = 'panel'; %#ok<SAGROW>
    if p <= nHinges
        segLengths(end+1) = L_hinge; segEA(end+1) = EA_hinge; segEI(end+1) = EI_hinge; segTypes{end+1} = 'hinge'; %#ok<SAGROW>
    end
end

nEdges = numel(segLengths);
nNodes = nEdges + 1;
nDOFs  = 2 * nNodes;
l_bar  = segLengths(:);

x = zeros(2, nNodes);
theta = 0.0;
for e = 1:nEdges
    if e > 1 && (e - 1) <= numel(phi_init)
        theta = theta + phi_init(e - 1);
    end
    x(:, e+1) = x(:, e) + segLengths(e) * [cos(theta); sin(theta)];
end

if strcmpi(rootBC, 'clamped')
    fixedNodes = [1, 2]; % x_0 and x_1 fixed (1-based indices)
else
    fixedNodes = 1;      % only x_0 fixed
end

constrainedDOFs = [];
for fn = fixedNodes
    constrainedDOFs = [constrainedDOFs, 2*fn - 1, 2*fn]; %#ok<AGROW>
end
constrainedDOFs = unique(constrainedDOFs);
freeDOFs = setdiff(1:nDOFs, constrainedDOFs);

% Energies at current configuration
E_s = 0.0;
edgeVecs = zeros(2, nEdges);
for e = 1:nEdges
    ev = x(:, e+1) - x(:, e);
    edgeVecs(:, e) = ev;
    strain = (norm(ev) / l_bar(e)) - 1.0;
    E_s = E_s + 0.5 * segEA(e) * (strain^2) * l_bar(e);
end

E_b = 0.0;
for i = 2:(nNodes - 1)
    ep = edgeVecs(:, i-1);
    en = edgeVecs(:, i);
    phi_i = atan2(ep(1)*en(2) - ep(2)*en(1), dot(ep, en));
    EI_v = min(segEI(i-1), segEI(i));
    dl_v = 0.5 * (l_bar(i-1) + l_bar(i));
    E_b = E_b + 0.5 * (EI_v / dl_v) * ((phi_i - phi_target(i-1))^2);
end
E_tot = E_s + E_b;

%% 3. Draw Directly in a MATLAB Figure Window
fig = figure('Name', 'MAE-263F Solar Array Physical Model', ...
    'Color', 'w', 'Units', 'inches', 'Position', [1.5, 1.2, 10.5, 6.8]);
ax  = axes('Parent', fig, 'Position', [0.07, 0.34, 0.89, 0.58]);
hold(ax, 'on'); axis(ax, 'equal'); grid(ax, 'on'); box(ax, 'on');

% Spacecraft bus wall
wallW = 0.14;
yLim  = 0.80;
rectangle(ax, 'Position', [-wallW, -yLim, wallW, 2*yLim], ...
    'FaceColor', [0.86 0.89 0.92], 'EdgeColor', [0.17 0.24 0.31], 'LineWidth', 1.4);
for yh = (-yLim + 0.12):0.16:(yLim - 0.06)
    plot(ax, [-wallW, 0], [yh - 0.09, yh], 'Color', [0.36 0.43 0.49], 'LineWidth', 0.85);
end
text(ax, -wallW - 0.06, 0, {'Spacecraft Bus', sprintf('(%s Root)', upper(rootBC))}, ...
    'HorizontalAlignment', 'right', 'VerticalAlignment', 'middle', ...
    'FontWeight', 'bold', 'FontSize', 9);

% Segments
hPanel = gobjects(1); hHinge = gobjects(1);
for e = 1:nEdges
    if strcmp(segTypes{e}, 'panel')
        hPanel = plot(ax, x(1,[e,e+1]), x(2,[e,e+1]), '-', ...
            'Color', [0.10 0.31 0.46], 'LineWidth', 4.2);
    else
        hHinge = plot(ax, x(1,[e,e+1]), x(2,[e,e+1]), '-', ...
            'Color', [0.83 0.33 0.00], 'LineWidth', 4.8);
    end
end

% Nodes and DOF labels
yOffsets = [0.16, -0.18, 0.18, -0.20, 0.18, 0.16];
for i = 1:nNodes
    d1 = 2*i - 1;
    d2 = 2*i;
    isFixed = ismember(d1, constrainedDOFs) && ismember(d2, constrainedDOFs);
    if isFixed
        plot(ax, x(1,i), x(2,i), 's', 'MarkerSize', 8.5, ...
            'MarkerFaceColor', [0.17 0.24 0.31], 'MarkerEdgeColor', 'w', 'LineWidth', 1.2);
        tag = 'FIXED';
        col = [0.48 0.14 0.11];
    else
        plot(ax, x(1,i), x(2,i), 'o', 'MarkerSize', 7.5, ...
            'MarkerFaceColor', 'w', 'MarkerEdgeColor', [0.10 0.31 0.46], 'LineWidth', 1.5);
        tag = 'FREE';
        col = [0.10 0.31 0.46];
    end
    dy = yOffsets(min(i, numel(yOffsets)));
    text(ax, x(1,i), x(2,i) + dy, ...
        sprintf('x_%d [%s]\nDOFs %d,%d', i-1, tag, d1, d2), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
        'FontSize', 9, 'FontWeight', 'bold', 'Color', col);
end

% Elastic & damping force arrows on free nodes
for i = 3:nNodes
    sgn = 1;
    if mod(i, 2) == 1, sgn = -1; end
    quiver(ax, x(1,i), x(2,i), 0.07, sgn*0.16, 0, ...
        'Color', [0.12 0.52 0.28], 'LineWidth', 1.3, 'MaxHeadSize', 0.8);
    quiver(ax, x(1,i), x(2,i), -0.06, -sgn*0.10, 0, ...
        'Color', [0.56 0.27 0.68], 'LineWidth', 1.2, 'LineStyle', '--', 'MaxHeadSize', 0.8);
end

xlabel(ax, 'x Position [m]', 'FontSize', 10);
ylabel(ax, 'y Position [m]', 'FontSize', 10);
title(ax, sprintf('Solar Array Sketch (N = %d Nodes, %d Total DOFs)', nNodes, nDOFs), ...
    'FontSize', 12, 'FontWeight', 'bold');
xlim(ax, [-0.95, 3.45]);
ylim(ax, [-0.88, 0.88]);

hFel = plot(ax, nan, nan, '-', 'Color', [0.12 0.52 0.28], 'LineWidth', 1.4);
hFdp = plot(ax, nan, nan, '--', 'Color', [0.56 0.27 0.68], 'LineWidth', 1.3);
legend(ax, [hPanel, hHinge, hFel, hFdp], ...
    {'Rigid Panel Segment', 'Compliant Hinge Segment', ...
    'Elastic Force F^{elastic}_i = -\nabla_{x_i} E_{total}', ...
    'Damping Force F^{damp}_i = -c \dot{x}_i'}, ...
    'Location', 'southeast', 'FontSize', 9);
% Summary text boxes drawn inside the MATLAB figure below the axes
boxTop = 0.22;
boxBg  = [0.97 0.98 0.99];
boxEc  = [0.36 0.43 0.49];

bcStr = { ...
    sprintf('Boundary Conditions (%s Root)', upper(rootBC)), ...
    sprintf('• Fixed Root: x_0 = [0,0], x_1 = [%.1f,0]', L_panel), ...
    sprintf('• Constrained DOFs: [%s]', num2str(constrainedDOFs)), ...
    sprintf('• Free Tip (Node %d): F_{ext} = 0, M_{tip} = 0', nNodes-1)};

dofStr = { ...
    'Degrees of Freedom (DOFs)', ...
    sprintf('• Total DOFs (2N): %d  (DOFs 1..%d)', nDOFs, nDOFs), ...
    sprintf('• Constrained (%d): [%s]', numel(constrainedDOFs), num2str(constrainedDOFs)), ...
    sprintf('• Free DOFs (%d): %d..%d', numel(freeDOFs), freeDOFs(1), freeDOFs(end))};

enStr = { ...
    'Forces & Energies (t = 0)', ...
    '• F^{elastic} = -\nabla E_{total}(q),  F^{damp} = -c v', ...
    sprintf('• Stretching E_s = %.3e J', E_s), ...
    sprintf('• Bending E_b = %.3f J | E_{total} = %.3f J', E_b, E_tot)};

annotation(fig, 'textbox', [0.04, boxTop-0.18, 0.29, 0.18], 'String', bcStr, ...
    'FitBoxToText', 'off', 'BackgroundColor', boxBg, 'EdgeColor', boxEc, 'FontSize', 9.5);
annotation(fig, 'textbox', [0.36, boxTop-0.18, 0.28, 0.18], 'String', dofStr, ...
    'FitBoxToText', 'off', 'BackgroundColor', boxBg, 'EdgeColor', boxEc, 'FontSize', 9.5);
annotation(fig, 'textbox', [0.67, boxTop-0.18, 0.29, 0.18], 'String', enStr, ...
    'FitBoxToText', 'off', 'BackgroundColor', boxBg, 'EdgeColor', boxEc, 'FontSize', 9.5);