﻿/* Copyright (C) 2020-2021 Artifex Software, Inc.
   All Rights Reserved.

   This software is provided AS-IS with no warranty, either express or
   implied.

   This software is distributed under license and may not be copied,
   modified or distributed except as expressly authorized under the terms
   of the license contained in the file LICENSE in this distribution.

   Refer to licensing information at http://www.artifex.com or contact
   Artifex Software, Inc.,  1305 Grant Avenue - Suite 200, Novato,
   CA 94945, U.S.A., +1(415)492-9861, for further information.
*/

using System;
using System.ComponentModel;
using System.Collections.ObjectModel;

namespace gs_mono_example
{
	public class DocPage : INotifyPropertyChanged
	{
		private int height;
		private int width;
		private double zoom;
        private Gdk.Pixbuf pixbuf;
        private String pagename;
		private int pagenum;
		private Page_Content_t content;

		public int Height
		{
			get { return height; }
			set
			{
				height = value;
				OnPropertyChanged("Height");
			}
		}

		public int Width
		{
			get { return width; }
			set
			{
				width = value;
				OnPropertyChanged("Width");
			}
		}

		public double Zoom
		{
			get { return zoom; }
			set { zoom = value; }
		}

		public Gdk.Pixbuf PixBuf
		{
			get { return pixbuf; }
			set
			{
                pixbuf = value;
				OnPropertyChanged("PixBuf");
			}
		}

		public String PageName
		{
			get { return pagename; }
			set { pagename = value; }
		}

		public int PageNum
		{
			get { return pagenum; }
			set { pagenum = value; }
		}
		public Page_Content_t Content
		{
			get { return content; }
			set { content = value; }
		}

		public event PropertyChangedEventHandler PropertyChanged;

		// Create the OnPropertyChanged method to raise the event
		protected void OnPropertyChanged(string name)
		{
			PropertyChangedEventHandler handler = PropertyChanged;
			if (handler != null)
			{
				handler(this, new PropertyChangedEventArgs(name));
			}
		}

		public DocPage()
		{
			this.height = 0;
			this.width = 0;
			this.zoom = 0;
			this.pixbuf = null;
			this.pagenum = -1;
			this.pagename = "";
		}

		public DocPage(int Height, int Width, double Zoom, Gdk.Pixbuf PixBuf, int PageNum)
		{
			this.height = Height;
			this.width = Width;
			this.zoom = Zoom;
			this.pixbuf = PixBuf;
			this.pagename = ("Page " + (pagenum + 1));
		}
	};
	public class Pages : ObservableCollection<DocPage>
	{
		public Pages()
			: base()
		{
		}
	}
}