from dependencies import *

from gui.TrainingPerformance import TrainingPerformance

class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Model Analysis")
        self.showFullScreen()
        
        # Central widget for managing different views
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)
        
        # Add views to the stacked widget
        self.scene_view = TrainingPerformance(parent = self)
        
        self.stacked_widget.addWidget(self.scene_view)
        
        # Create and configure the toolbar
        self.toolbar = QToolBar("Main Toolbar")
        self.addToolBar(Qt.ToolBarArea.BottomToolBarArea, self.toolbar)
        self.toolbar.setMovable(False)
        self.toolbar.setFloatable(False)
        
        # Add view buttons to the left of the toolbar
        scene_btn = QPushButton('Training Performance')
        scene_btn.clicked.connect(lambda : self.stacked_widget.setCurrentIndex(0))

        # Use QWidgetAction to align buttons
        left_spacer = QWidget()
        left_spacer.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )

        exit_btn = QPushButton("Exit")
        exit_btn.clicked.connect(self.close)
        
        self.toolbar.addWidget(scene_btn)
        self.toolbar.addWidget(left_spacer)
        self.toolbar.addWidget(exit_btn)

    def update_(self):
        for page in range(self.stacked_widget.count()):
            self.stacked_widget.widget(page).update_()

    def next_page(self):
        self.stacked_widget.setCurrentIndex(self.stacked_widget.currentIndex() + 1)
        self.update_()
        
    def prev_page(self):
        self.stacked_widget.setCurrentIndex(self.stacked_widget.currentIndex() - 1)
        self.update_()

    def goto_page(self, page_num):
        self.stacked_widget.setCurrentIndex(page_num)
        self.update_()

    def show_notification( self, msg: str ):
        popup = QMessageBox( self )
        popup.setWindowTitle( "Notification" )
        popup.setText( msg )
        popup.setIcon( QMessageBox.Icon.Information )
        popup.setStandardButtons( QMessageBox.StandardButton.Ok )

        response = popup.exec()

    def get_left_toolbar_layout( self, child, title: str ):
        '''
        Returns a standard layout with a left toolbar (1/6 of the window) and a main window area (5/6 of the window).

        @returns
            full_window (QHBoxLayout; child of child)
            left_toolbar (QVBoxLayout; child of full_window
            main_area (QVBoxLayout; child of full_window)
        '''

        full_window = QHBoxLayout( child )

        # Build main layouts
        left_area = QVBoxLayout( self )
        main_area = QVBoxLayout( self )

        # Build separator line
        sep_line = QFrame()
        sep_line.setFrameShape( QFrame.Shape.VLine )

        full_window.addLayout( left_area, 5 )
        full_window.addWidget( sep_line, 1 )
        full_window.addLayout( main_area, 55 )

        # Build left toolbar
        title = QLabel( title )
        title.setFont( QFont('Arial', 24) )
        title.setAlignment( Qt.AlignmentFlag.AlignCenter )
        left_area.addWidget( title, 5 )

        # Build separator line
        h_line = QFrame()
        h_line.setFrameShape( QFrame.Shape.HLine )
        left_area.addWidget( h_line, 1 )

        # Build left_toolbar
        left_toolbar = QVBoxLayout()
        left_area.addLayout( left_toolbar, 15 )

        return full_window, left_toolbar, main_area