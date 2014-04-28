function displaySymbolicLatex( symbol )

latexSymbol = latex(symbol);

fig = figure;
set( fig, 'Units', 'pixels', 'Menubar', 'none' );
set( gca, 'Box', 'off', 'XTick', [], 'YTick', [], 'Units', 'pixels' );

t = text( 0, 0, ['$$' latexSymbol '$$'], 'interpreter', 'latex', 'FontSize', 14 );
set( t, 'Units', 'pixels' );
extent = get( t, 'Extent' );

offset = 50;
position = [500 500 extent(3:4)+offset];
set( fig, 'Position', position );
set( gca, 'Position', [0 0 position(3:4)] );
set( t, 'Position', [(position(3)-extent(3))/2 position(4)/2] );

shg;